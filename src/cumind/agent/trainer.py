"""Training loop implementation."""

import sys
from typing import Any, Dict, List, Tuple

import chex
import jax
import jax.numpy as jnp
import optax  # type: ignore
from flax import nnx
from tqdm import tqdm  # type: ignore

from cumind.agent.agent import Agent
from cumind.core.network import CuMindNetwork
from cumind.data.memory import Memory
from cumind.data.self_play import SelfPlay
from cumind.utils.checkpoint import load_checkpoint, save_checkpoint
from cumind.utils.config import cfg
from cumind.utils.jax_utils import pmap_n_step_return, vmap_n_step_return
from cumind.utils.logger import log


class TqdmSink:
    def __init__(self, mode: bool):
        self.mode = mode
        if mode:
            self.sink = self._stdout_sink
        else:
            self.sink = self._logger_sink

    def write(self, msg: Any) -> None:
        self.sink(msg)

    def _stdout_sink(self, msg: Any) -> None:
        sys.stdout.write(str(msg))

    def _logger_sink(self, msg: Any) -> None:
        log.info(str(msg))


class Trainer:
    """Orchestrates the training process, including sampling, updates, and logging."""

    def __init__(self, agent: Agent, memory: Memory):
        """Initializes the Trainer.
        Args:
            agent: The agent to train.
            memory: The memory buffer for sampling training data.
        """
        log.info(f"Initializing trainer for environment: {cfg.env.name}")
        self.agent = agent
        self.memory = memory

        self.checkpoint_dir = log.get_checkpoint_dir()
        log.info(f"Checkpoints will be saved to {self.checkpoint_dir}")
        self.step_count = 0
        self.last_loss: Dict[str, float] = {}

    def train(self, env: Any) -> None:
        """Runs the main training loop."""
        num_episodes = cfg.training.num_episodes
        train_frequency = cfg.training.train_frequency
        tqdm_file = TqdmSink(cfg.logging.tqdm)
        pbar = tqdm(range(1, num_episodes + 1), desc="Training Progress", file=tqdm_file)
        self_play = SelfPlay(self.agent, self.memory)
        for episode in pbar:
            reward, steps, _ = self_play.run_episode(env)
            self.log_metrics(episode, reward, steps)
            if episode % train_frequency == 0:
                self.last_loss = self.step()
                if self.step_count % cfg.training.target_update_frequency == 0:
                    self.agent.update_target_network()
            if episode % cfg.training.checkpoint_interval == 0:
                self.checkpoint(episode)

    def step(self) -> Dict[str, float]:
        """Performs one full training step, including sampling and network update."""
        if not self.memory.is_ready(cfg.memory.min_size, cfg.memory.min_pct):
            log.warning("Buffer not ready for training, skipping step.")
            return {}
        log.debug(f"Starting training step {self.step_count}...")
        batch = self.memory.sample(cfg.training.batch_size)
        obs, acts, targets = self.prepare_batch(batch)

        params = nnx.state(self.agent.network, nnx.Param)

        @jax.jit
        def train_step(params: nnx.State[Any, Any], opt_state: Any, obs: chex.Array, acts: chex.Array, targets: Dict[str, chex.Array]) -> Tuple[chex.Array, Dict[str, chex.Array], Any, nnx.State[Any, Any]]:
            grad_fn = jax.value_and_grad(self.loss, has_aux=True)
            (total_loss, losses), grads = grad_fn(params, obs, acts, targets)
            updates, opt_state = self.agent.optimizer.update(grads, opt_state, params)
            updated_params = optax.apply_updates(params, updates)
            updated_state = nnx.state(updated_params, nnx.Param)
            return total_loss, losses, opt_state, updated_state

        total_loss, losses, self.agent.optimizer_state, updated_params = train_step(params, self.agent.optimizer_state, obs, acts, targets)
        nnx.update(self.agent.network, updated_params)
        log.debug(f"Training step {self.step_count} complete.")

        losses_float = {f"train/{k}": float(v) for k, v in losses.items()}
        losses_float["total_loss"] = float(total_loss)
        log.log_scalars(losses_float, self.step_count)
        self.step_count += 1
        return {"total_loss": float(total_loss), **losses_float}

    def loss(self, params: nnx.State[Any, Any], obs: chex.Array, acts: chex.Array, targets: Dict[str, chex.Array]) -> Tuple[chex.Array, Dict[str, chex.Array]]:
        """Computes the total loss for a batch."""
        net = nnx.clone(self.agent.network)
        nnx.update(net, params)

        losses = self.compute_losses(net, obs, acts, targets)
        total = jnp.sum(jnp.array(list(losses.values())))
        return total, losses

    def prepare_batch(self, batch: List[Any]) -> Tuple[chex.Array, chex.Array, Dict[str, chex.Array]]:
        """Prepares a batch of trajectories for training."""
        obs, acts, policies, rewards = [], [], [], []
        value_inputs_rewards, value_inputs_values = [], []
        n = cfg.selfplay.td_steps
        d = cfg.selfplay.discount
        for item in batch:
            if not item:
                raise RuntimeError("Empty item in batch.")
            policies.append(item[0]["policy"])
            obs.append(item[0]["observation"])
            acts.append([step["action"] for step in item[: cfg.selfplay.num_unroll_steps]])
            rewards_seq = [step["reward"] for step in item[: cfg.selfplay.num_unroll_steps]]
            rewards.append(rewards_seq)
            value_inputs_rewards.append(jnp.array([step["reward"] for step in item], dtype=jnp.float32))
            if len(item) > n:
                last_obs = jnp.array(item[n]["observation"])[None, :]
                _, _, value = self.agent.network.initial_inference(last_obs, use_target=True)
                value_inputs_values.append(jnp.array([float(jnp.asarray(value)[0, 0])], dtype=jnp.float32))
            else:
                value_inputs_values.append(jnp.array([0.0], dtype=jnp.float32))
        acts = [a + [0] * (cfg.selfplay.num_unroll_steps - len(a)) for a in acts]
        rewards = [r + [0.0] * (cfg.selfplay.num_unroll_steps - len(r)) for r in rewards]
        if cfg.multi_device:
            value_targets = pmap_n_step_return(jnp.stack(value_inputs_rewards), jnp.stack(value_inputs_values), n, d)
        else:
            value_targets = vmap_n_step_return(jnp.stack(value_inputs_rewards), jnp.stack(value_inputs_values), n, d)
        return (
            jnp.array(obs),
            jnp.array(acts, dtype=jnp.int32),
            {
                "values": value_targets,
                "rewards": jnp.array(rewards, dtype=jnp.float32),
                "policies": jnp.array(policies, dtype=jnp.float32),
            },
        )

    def n_step_return(self, item: List[Dict[str, Any]]) -> float:
        """Computes the n-step return for a item, with bootstrapping."""
        n = cfg.selfplay.td_steps
        d = cfg.selfplay.discount
        rewards = jnp.array([step["reward"] for step in item], dtype=jnp.float32)
        if len(item) > n:
            last_obs = jnp.array(item[n]["observation"])[None, :]
            _, _, value = self.agent.network.initial_inference(last_obs, use_target=True)
            boot = float(jnp.asarray(value)[0, 0])
        else:
            boot = 0.0
        idx = jnp.arange(n)
        mask = idx < rewards.shape[0]
        rewards_pad = jnp.where(mask, rewards[:n], 0)
        discounts = d**idx
        ret = jnp.sum(rewards_pad * discounts)
        if boot != 0.0:
            ret += (d**n) * boot
        return float(ret)

    def compute_losses(self, net: CuMindNetwork, obs: chex.Array, acts: chex.Array, targets: Dict[str, chex.Array]) -> Dict[str, chex.Array]:
        """Computes the value, policy, and reward losses."""
        h, pol_logits, vals = net.initial_inference(obs)

        v_loss = jnp.mean((jnp.asarray(vals).squeeze() - jnp.asarray(targets["values"])) ** 2)
        p_loss = -jnp.mean(jnp.sum(jnp.asarray(targets["policies"]) * jax.nn.log_softmax(pol_logits, axis=-1), axis=-1))

        r_loss = jnp.array(0.0)

        def unroll(
            carry: chex.Array,
            step: chex.Array,
        ) -> Tuple[chex.Array, Tuple[chex.Array, chex.Array, chex.Array]]:
            s = carry
            a = jnp.asarray(acts)[:, step]
            ns, pr, ppl, pv = net.recurrent_inference(s, a)

            pr = jnp.asarray(pr).squeeze()
            tr = jnp.asarray(targets["rewards"])[:, step]
            rl = jnp.mean((pr - tr) ** 2)

            pv = jnp.asarray(pv).squeeze()
            vl = jnp.mean((pv - jnp.asarray(targets["values"])) ** 2)

            plp = jax.nn.log_softmax(ppl, axis=-1)
            pl = -jnp.mean(jnp.sum(jnp.asarray(targets["policies"]) * plp, axis=-1))

            return ns, (rl, vl, pl)

        _, losses = jax.lax.scan(unroll, h, jnp.arange(cfg.selfplay.num_unroll_steps))
        r_loss += jnp.mean(losses[0])
        v_loss += jnp.mean(losses[1])
        p_loss += jnp.mean(losses[2])
        if cfg.selfplay.num_unroll_steps > 0:
            r_loss /= cfg.selfplay.num_unroll_steps
            v_loss /= cfg.selfplay.num_unroll_steps + 1
            p_loss /= cfg.selfplay.num_unroll_steps + 1

        return {"value_loss": v_loss, "policy_loss": p_loss, "reward_loss": r_loss}

    def log_metrics(self, episode: int, reward: float, steps: int) -> None:
        log.info(f"Episode {episode:3d}: Reward={reward:6.1f}, Length={steps:3d}, Loss={self.last_loss.get('total_loss', 0):.4f}, Memory={self.memory.get_pct():2.2f}")

    def checkpoint(self, episode: int) -> None:
        """Saves the agent's state to a checkpoint file."""
        state = self.agent.save_state()
        path = f"{self.checkpoint_dir}/episode_{episode:05d}.pkl"
        save_checkpoint(state, path)

    def load(self, path: str) -> None:
        """Loads the agent's state from a checkpoint file."""
        log.info(f"Loading checkpoint from {path}")
        state = load_checkpoint(path)
        self.agent.load_state(state)
        log.info("Checkpoint loaded successfully.")
