"""Training loop implementation."""

from typing import Any, Dict, List, Optional, Tuple

import pickle
import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax  # type: ignore
from flax import nnx
from tqdm import tqdm  # type: ignore

from cumind.agent.agent import Agent
from cumind.core.network import CuMindNetwork
from cumind.data.memory import Memory
from cumind.data.self_play import SelfPlay
from cumind.utils.checkpoint import (
    AgentState,
    CheckpointData,
    CheckpointMetadata,
    load_checkpoint,
    save_checkpoint,
)
from cumind.utils.config import cfg
from cumind.utils.jax_utils import pmap_n_step_return, vmap_n_step_return
from cumind.utils.logger import TqdmSink, log


# This entire block of computation will be JIT-compiled and run on the GPU.
def _train_step_impl(
    network: CuMindNetwork,
    optimizer: optax.GradientTransformation,
    params: nnx.State[Any, Any],
    opt_state: optax.OptState,
    observations: chex.Array,
    actions: chex.Array,
    policy_targets: chex.Array,
    reward_targets: chex.Array,
    bootstrap_obs: chex.Array,
    rewards_stack: chex.Array,
    bootstrap_mask: chex.Array,
) -> Tuple[chex.Array, Dict[str, chex.Array], Any, optax.OptState]:
    """Core training step implementation (to be JIT-compiled)."""

    # 1. Calculate n-step returns entirely on the device
    _, _, bootstrap_values = network.initial_inference(bootstrap_obs, use_target=True)
    bootstrap_values = jnp.asarray(bootstrap_values).squeeze() * bootstrap_mask

    # Construct the full values stack for the n-step return calculation
    values_stack = jnp.zeros_like(rewards_stack).at[:, cfg.selfplay.td_steps].set(bootstrap_values)

    # Compute the final value targets
    if cfg.multi_device:
        value_targets = pmap_n_step_return(jnp.asarray(rewards_stack), jnp.asarray(values_stack), cfg.selfplay.td_steps, cfg.selfplay.discount)
    else:
        value_targets = vmap_n_step_return(jnp.asarray(rewards_stack), jnp.asarray(values_stack), cfg.selfplay.td_steps, cfg.selfplay.discount)

    targets = {
        "values": value_targets,
        "rewards": reward_targets,
        "policies": policy_targets,
    }

    # 2. Compute loss and gradients
    def loss_fn(params: nnx.State[Any, Any]) -> Tuple[chex.Array, Dict[str, chex.Array]]:
        temp_network = nnx.clone(network)
        nnx.update(temp_network, params)
        losses = _compute_losses(temp_network, observations, actions, targets)
        total_loss = jnp.sum(jnp.array(list(losses.values())))
        return total_loss, losses

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (total_loss, losses), grads = grad_fn(params)

    # 3. Apply updates
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    return total_loss, losses, new_params, new_opt_state


def _compute_losses(network: CuMindNetwork, observations: chex.Array, actions: chex.Array, targets: Dict[str, chex.Array]) -> Dict[str, chex.Array]:
    """Computes the value, policy, and reward losses."""
    hidden_states, initial_policy_logits, initial_values = network.initial_inference(observations)
    value_loss = jnp.mean((jnp.asarray(initial_values).squeeze() - jnp.asarray(targets["values"])) ** 2)
    policy_loss = -jnp.mean(jnp.sum(jnp.asarray(targets["policies"]) * jax.nn.log_softmax(initial_policy_logits, axis=-1), axis=-1))
    reward_loss = jnp.array(0.0)
    current_states = hidden_states
    for step in range(cfg.selfplay.num_unroll_steps):
        step_actions = jnp.asarray(actions)[:, step]
        next_states, pred_rewards, pred_policy_logits, pred_values = network.recurrent_inference(current_states, step_actions)
        pred_rewards_squeezed = jnp.asarray(pred_rewards).squeeze()
        target_rewards = jnp.asarray(targets["rewards"])[:, step]
        reward_loss += jnp.mean((pred_rewards_squeezed - target_rewards) ** 2)
        pred_values_squeezed = jnp.asarray(pred_values).squeeze()
        value_loss += jnp.mean((pred_values_squeezed - jnp.asarray(targets["values"])) ** 2)
        policy_log_probs = jax.nn.log_softmax(pred_policy_logits, axis=-1)
        policy_loss += -jnp.mean(jnp.sum(jnp.asarray(targets["policies"]) * policy_log_probs, axis=-1))
        current_states = next_states
    if cfg.selfplay.num_unroll_steps > 0:
        reward_loss /= cfg.selfplay.num_unroll_steps
        value_loss /= cfg.selfplay.num_unroll_steps + 1
        policy_loss /= cfg.selfplay.num_unroll_steps + 1
    return {"value_loss": value_loss, "policy_loss": policy_loss, "reward_loss": reward_loss}


_jitted_train_step = jax.jit(_train_step_impl, static_argnames=["network", "optimizer"])


class Trainer:
    """Orchestrates the training process."""

    def __init__(self, agent: Agent, memory: Memory):
        log.info(f"Initializing trainer for environment: {cfg.env.name}")
        self.agent = agent
        self.memory = memory
        self.checkpoint_dir = log.get_checkpoint_dir()
        log.info(f"Checkpoints will be saved to {self.checkpoint_dir}")
        self.train_step_count = 0
        self.start_episode = 1  # Track starting episode for resumption

    def train(self, env: Any, resume_from_checkpoint: Optional[str] = None) -> None:
        # Resume from checkpoint if provided
        if resume_from_checkpoint:
            self._resume_from_checkpoint(resume_from_checkpoint)
        
        num_episodes = cfg.training.num_episodes
        train_frequency = cfg.training.train_frequency
        tqdm_file = TqdmSink(cfg.logging.tqdm)
        pbar = tqdm(range(self.start_episode, num_episodes + 1), desc="Training Progress", file=tqdm_file)
        self_play = SelfPlay(self.agent, self.memory)
        self.last_loss: Dict[str, float] = {}
        for episode in pbar:
            self._run_episode_and_log(env, self_play, episode)
            self._maybe_train_and_update(episode, train_frequency)
            self._maybe_checkpoint(episode)

    def _resume_from_checkpoint(self, checkpoint_path: str) -> None:
        """Resume training from a checkpoint."""
        log.info(f"Resuming training from checkpoint: {checkpoint_path}")
        
        try:
            # Load the full checkpoint data to access both state and metadata
            checkpoint_data: CheckpointData = load_checkpoint(checkpoint_path)
            
            self.agent.load_state(checkpoint_data['state'])
            
            metadata = checkpoint_data.get('metadata', {})
            self.train_step_count = metadata.get('train_step_count', 0)
            self.start_episode = metadata.get('episode', 0) + 1
            self.last_loss = metadata.get('last_loss', {})
            
            log.info(f"Resumed from episode {metadata.get('episode', 0)}, training step {self.train_step_count}")
            
        except Exception as e:
            log.exception(f"Failed to resume from checkpoint {checkpoint_path}: {e}")
            raise

    def _run_episode_and_log(self, env: Any, self_play: SelfPlay, episode: int) -> None:
        episode_reward, episode_steps, _ = self_play.run_episode(env)
        metrics = {
            "Episode": episode,
            "Reward": float(episode_reward),
            "Length": episode_steps,
            "Loss": float(self.last_loss.get("total_loss", 0)),
            "Memory": float(self.memory.get_pct()),
        }
        self._log_metrics(metrics)

    def _maybe_train_and_update(self, episode: int, train_frequency: int) -> None:
        if episode > 0 and episode % train_frequency == 0:
            if not self.memory.is_ready(cfg.memory.min_size, cfg.memory.min_pct):
                log.warning("Buffer not ready for training, a larger buffer is needed.")
                return
            for _ in range(cfg.training.num_batches):
                self.last_loss = self.train_step()
                self.train_step_count += 1
                if self.train_step_count > 0 and self.train_step_count % cfg.training.target_update_frequency == 0:
                    log.info(f"Updating target network at training step {self.train_step_count}")
                    self.agent.update_target_network()
                    log.info("Target network update completed")

    def _maybe_checkpoint(self, episode: int) -> None:
        if episode > 0 and episode % cfg.training.checkpoint_interval == 0:
            state: AgentState = self.agent.save_state()
            path = f"{self.checkpoint_dir}/episode_{episode:05d}.pkl"
            
            metadata: CheckpointMetadata = {
                "episode": episode,
                "train_step_count": self.train_step_count,
                "last_loss": self.last_loss,
                "memory_size": len(self.memory),
            }
            
            save_checkpoint(state, path, metadata)
            log.info(f"Checkpoint saved to {path}")

    def _log_metrics(self, metrics: Dict[str, Any]) -> None:
        log.info(f"Episode {metrics['Episode']:3d}: Reward={metrics['Reward']:6.1f}, Length={metrics['Length']:3d}, Loss={metrics['Loss']:.4f}, Memory={metrics['Memory']:2.2f}")

    def train_step(self) -> Dict[str, float]:
        log.debug(f"Starting training step {self.train_step_count}...")
        batch = self.memory.sample(cfg.training.batch_size)
        (
            observations,
            actions,
            policy_targets,
            reward_targets,
            bootstrap_obs,
            rewards_stack,
            bootstrap_mask,
        ) = self._prepare_batch(batch)

        params = nnx.state(self.agent.network, nnx.Param)
        total_loss, losses, new_params, new_opt_state = _jitted_train_step(
            self.agent.network,
            self.agent.optimizer,
            params,
            self.agent.optimizer_state,
            observations,
            actions,
            policy_targets,
            reward_targets,
            bootstrap_obs,
            rewards_stack,
            bootstrap_mask,
        )
        self.agent.optimizer_state = new_opt_state
        nnx.update(self.agent.network, new_params)
        log.debug(f"Training step {self.train_step_count} complete.")
        losses_float = {f"train/{k}": float(v) for k, v in losses.items()}
        losses_float["total_loss"] = float(total_loss)
        log.log_scalars(losses_float, self.train_step_count)
        return {"total_loss": float(total_loss), **losses_float}

    def _prepare_batch(self, batch: List[Any]) -> Tuple[np.ndarray, ...]:
        """Prepares a batch for training with pure NumPy, keeping it off the GPU."""
        batch_size = len(batch)
        unroll_steps = cfg.selfplay.num_unroll_steps
        n_steps = cfg.selfplay.td_steps
        obs_shape = cfg.env.observation_shape
        max_len = cfg.env.max_episode_steps

        observations = np.zeros((batch_size, *obs_shape), dtype=np.float32)
        policy_targets = np.zeros((batch_size, cfg.env.action_space_size), dtype=np.float32)
        action_sequences = np.zeros((batch_size, unroll_steps), dtype=np.int32)
        reward_targets = np.zeros((batch_size, unroll_steps), dtype=np.float32)
        rewards_stack = np.zeros((batch_size, max_len), dtype=np.float32)
        bootstrap_obs_batch = np.zeros((batch_size, *obs_shape), dtype=np.float32)
        bootstrap_mask = np.zeros((batch_size,), dtype=np.bool_)

        for i, item in enumerate(batch):
            item_len = len(item)
            observations[i] = item[0]["observation"]
            policy_targets[i] = item[0]["policy"]
            actions = [step["action"] for step in item[:unroll_steps]]
            action_sequences[i, : len(actions)] = actions
            full_rewards = [step["reward"] for step in item]
            rewards_stack[i, :item_len] = full_rewards
            unroll_rewards = full_rewards[:unroll_steps]
            reward_targets[i, : len(unroll_rewards)] = unroll_rewards
            if item_len > n_steps:
                bootstrap_obs_batch[i] = item[n_steps]["observation"]
                bootstrap_mask[i] = True

        return (
            observations,
            action_sequences,
            policy_targets,
            reward_targets,
            bootstrap_obs_batch,
            rewards_stack,
            bootstrap_mask,
        )

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Load checkpoint and return metadata for training resumption."""
        state: AgentState = load_checkpoint(path)["state"]
        self.agent.load_state(state)
        return {"loaded_from": path, **state.get("metadata", {})}
