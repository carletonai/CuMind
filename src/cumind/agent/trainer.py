"""Training loop implementation."""

import os
from datetime import datetime
from typing import Any, Dict, List, Tuple

import chex
import jax
import jax.numpy as jnp
import optax  # type: ignore
from flax import nnx
from tqdm import tqdm  # type: ignore

from ..config import Config
from ..core.network import CuMindNetwork
from ..data.memory import Memory
from ..data.self_play import SelfPlay
from ..utils.checkpoint import load_checkpoint, save_checkpoint
from ..utils.logger import log
from .agent import Agent


class Trainer:
    """Orchestrates the training process, including sampling, updates, and logging."""

    def __init__(self, agent: Agent, memory: Memory, config: Config):
        """Initializes the Trainer.
        Args:
            agent: The agent to train.
            buffer: The memory buffer for sampling training data.
            config: The configuration object.
            env_name: The name for the environment, used for creating checkpoint directories.
        """
        log.info(f"Initializing trainer for environment: {config.env_name}")
        self.agent = agent
        self.memory = memory
        self.config = config
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.checkpoint_dir = f"{config.checkpoint_root_dir}/{config.env_name}/{timestamp}"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        log.info(f"Checkpoints will be saved to {self.checkpoint_dir}")
        self.train_step_count = 0

    def run_training_loop(self, env: Any) -> None:
        """Runs the main training loop.
        Args:
            env: The environment to run episodes in.
        """
        num_episodes = self.config.num_episodes
        train_frequency = self.config.train_frequency

        log.info(f"Starting training loop for {num_episodes} episodes with train frequency {train_frequency}.")
        loss_info = {}
        pbar = tqdm(range(1, num_episodes + 1), desc="Training Progress")
        self_play = SelfPlay(self.config, self.agent, self.memory)
        for episode in pbar:
            episode_reward, episode_steps, _ = self_play.run_episode(env)


            if episode > 0 and episode % train_frequency == 0:
                loss_info = self.train_step()

                if self.train_step_count > 0 and self.train_step_count % self.config.target_update_frequency == 0:
                    log.info(f"Updating target network at training step {self.train_step_count}")
                    self.agent.update_target_network()
            metrics = {
                "Episode": episode,
                "Reward": float(episode_reward),
                "Length": episode_steps,
                "Loss": float(loss_info.get("total_loss", 0)),
                "Memory": float(self.memory.get_pct()),
            }
            log.info(
                f"Episode {metrics['Episode']:3d}: Reward={metrics['Reward']:6.1f}, "
                f"Length={metrics['Length']:3d}, Loss={metrics['Loss']:.4f}, "
                f"Memory={metrics['Memory']:2.2f}"
            )
            pbar.set_postfix(metrics)

            if episode > 0 and episode % self.config.checkpoint_interval == 0:
                self.save_checkpoint(episode)

    def train_step(self) -> Dict[str, float]:
        """Performs one full training step, including sampling and network update."""
        if not self.memory.is_ready(self.config.min_memory_size, self.config.min_memory_pct):
            log.warning("Buffer not ready for training, skipping step.")
            return {}
        log.debug(f"Starting training step {self.train_step_count}...")
        batch = self.memory.sample(self.config.batch_size)
        observations, actions, targets = self._prepare_batch(batch)

        params = nnx.state(self.agent.network, nnx.Param)

        grad_fn = jax.value_and_grad(self._loss_fn, has_aux=True)
        (total_loss, losses), grads = grad_fn(params, observations, actions, targets)

        updates, self.agent.optimizer_state = self.agent.optimizer.update(grads, self.agent.optimizer_state, params)
        updated_params = optax.apply_updates(params, updates)

        nnx.update(self.agent.network, updated_params)
        log.debug(f"Training step {self.train_step_count} complete.")

        losses_float = {f"train/{k}": float(v) for k, v in losses.items()}
        losses_float["total_loss"] = float(total_loss)
        log.log_scalars(losses_float, self.train_step_count)
        self.train_step_count += 1
        return {"total_loss": float(total_loss), **losses_float}

    def _loss_fn(self, params: nnx.State[Any, Any], observations: chex.Array, actions: chex.Array, targets: Dict[str, chex.Array]) -> Tuple[chex.Array, Dict[str, chex.Array]]:
        """Computes the total loss for a batch."""
        temp_network = nnx.clone(self.agent.network)
        nnx.update(temp_network, params)

        losses = self._compute_losses(temp_network, observations, actions, targets)
        total_loss = jnp.sum(jnp.array(list(losses.values())))
        return total_loss, losses

    def _prepare_batch(self, batch: List[Any]) -> Tuple[chex.Array, chex.Array, Dict[str, chex.Array]]:
        """Prepares a batch of trajectories for training."""
        observations, action_sequences, policy_targets, value_targets, reward_targets = [], [], [], [], []

        for item in batch:
            if not item:
                log.critical("Encountered empty item in batch.")
                raise RuntimeError("Encountered empty item in batch.")

            value = self._compute_n_step_return(item)
            value_targets.append(value)

            policy_targets.append(item[0]["policy"])
            observations.append(item[0]["observation"])

            actions = [step["action"] for step in item[: self.config.num_unroll_steps]]
            rewards = [step["reward"] for step in item[: self.config.num_unroll_steps]]
            action_sequences.append(actions)
            reward_targets.append(rewards)

        for seq in action_sequences:
            while len(seq) < self.config.num_unroll_steps:
                seq.append(0)
        for seq in reward_targets:
            while len(seq) < self.config.num_unroll_steps:
                seq.append(0.0)

        return (
            jnp.array(observations),
            jnp.array(action_sequences, dtype=jnp.int32),
            {
                "values": jnp.array(value_targets, dtype=jnp.float32),
                "rewards": jnp.array(reward_targets, dtype=jnp.float32),
                "policies": jnp.array(policy_targets, dtype=jnp.float32),
            },
        )

    def _compute_n_step_return(self, item: List[Dict[str, Any]]) -> float:
        """Computes the n-step return for a item, with bootstrapping."""
        n_steps = self.config.td_steps
        discount = self.config.discount
        rewards = [step["reward"] for step in item]
        n_step_return = 0.0

        for i in range(min(len(rewards), n_steps)):
            n_step_return += rewards[i] * (discount**i)

        if len(item) > n_steps:
            last_obs = jnp.array(item[n_steps]["observation"])[None, :]
            _, _, value = self.agent.target_network.initial_inference(last_obs)
            n_step_return += (discount**n_steps) * float(jnp.asarray(value)[0, 0])

        return n_step_return

    def _compute_losses(self, network: CuMindNetwork, observations: chex.Array, actions: chex.Array, targets: Dict[str, chex.Array]) -> Dict[str, chex.Array]:
        """Computes the value, policy, and reward losses."""
        hidden_states, initial_policy_logits, initial_values = network.initial_inference(observations)

        value_loss = jnp.mean((jnp.asarray(initial_values).squeeze() - jnp.asarray(targets["values"])) ** 2)
        policy_loss = -jnp.mean(jnp.sum(jnp.asarray(targets["policies"]) * jax.nn.log_softmax(initial_policy_logits, axis=-1), axis=-1))

        reward_loss = jnp.array(0.0)
        current_states = hidden_states

        # Accumulate value loss, policy loss, and reward loss for each unroll step
        for step in range(self.config.num_unroll_steps):
            step_actions = jnp.asarray(actions)[:, step]
            next_states, pred_rewards, pred_policy_logits, pred_values = network.recurrent_inference(current_states, step_actions)

            pred_rewards_squeezed = jnp.asarray(pred_rewards).squeeze()
            target_rewards = jnp.asarray(targets["rewards"])[:, step]
            reward_loss += jnp.mean((pred_rewards_squeezed - target_rewards) ** 2)

            pred_values_squeezed = jnp.asarray(pred_values).squeeze()
            # Note: A more advanced implementation might use a different target for unrolled steps
            value_loss += jnp.mean((pred_values_squeezed - jnp.asarray(targets["values"])) ** 2)

            policy_log_probs = jax.nn.log_softmax(pred_policy_logits, axis=-1)
            policy_loss += -jnp.mean(jnp.sum(jnp.asarray(targets["policies"]) * policy_log_probs, axis=-1))

            current_states = next_states

        if self.config.num_unroll_steps > 0:
            reward_loss /= self.config.num_unroll_steps
            value_loss /= self.config.num_unroll_steps + 1
            policy_loss /= self.config.num_unroll_steps + 1

        return {"value_loss": value_loss, "policy_loss": policy_loss, "reward_loss": reward_loss}

    def save_checkpoint(self, episode: int) -> None:
        """Saves the agent's state to a checkpoint file."""
        state = self.agent.save_state()
        path = f"{self.checkpoint_dir}/episode_{episode:05d}.pkl"
        log.info(f"Saving checkpoint at episode {episode} to {path}")
        save_checkpoint(state, path)

    def load_checkpoint(self, path: str) -> None:
        """Loads the agent's state from a checkpoint file."""
        log.info(f"Loading checkpoint from {path}")
        state = load_checkpoint(path)
        self.agent.load_state(state)
        log.info("Checkpoint loaded successfully.")
