"""Training loop implementation."""

from typing import Any, Dict, List, Tuple

import chex
import jax
import jax.numpy as jnp
import optax
from flax import nnx
from tqdm import tqdm

from ..config import Config
from ..core.network import CuMindNetwork
from ..data.memory_buffer import MemoryBuffer
from ..data.self_play import SelfPlay
from ..utils.checkpoint import load_checkpoint, save_checkpoint
from ..utils.logger import Logger
from .agent import Agent


class Trainer:
    """Orchestrates the training process, including sampling, updates, and logging."""

    def __init__(self, agent: Agent, buffer: MemoryBuffer, config: Config, logger: Logger):
        """Initializes the Trainer.
        Args:
            agent: The agent to train.
            buffer: The replay buffer for sampling training data.
            config: The configuration object.
            logger: The logger for recording metrics.
        """
        self.agent = agent
        self.buffer = buffer
        self.config = config
        self.logger = logger
        self.train_step_count = 0

    def run_training_loop(self, env: Any, num_episodes: int, train_frequency: int) -> None:
        """Runs the main training loop.
        Args:
            env: The environment to run episodes in.
            num_episodes: The total number of episodes to run.
            train_frequency: The number of episodes between training steps.
        """
        self.logger.log_text(f"Starting training loop for {num_episodes} episodes with train frequency {train_frequency}.")
        loss_info = {}
        pbar = tqdm(range(num_episodes), desc="Training Progress")
        for episode in pbar:
            self_play = SelfPlay(self.config, self.agent, self.buffer)
            episode_reward, episode_steps, _ = self_play.run_episode(env)

            self.logger.log_text(f"Episode {episode:3d}: Reward={episode_reward:6.1f}, Length={episode_steps:3d}")

            if episode % train_frequency == 0:
                loss_info = self.train_step()

            pbar.set_postfix(
                {
                    "Episode": episode,
                    "Reward": f"{episode_reward:.2f}",
                    "Length": episode_steps,
                    "Loss": f"{loss_info.get('total_loss', 0):.4f}",
                    "Memory": f"{self.buffer.get_pct():2.2f}",
                }
            )

            if episode % 50 == 0 and episode > 0:
                self.save_checkpoint(f"checkpoints/cartpole_episode_{episode}.pkl")

    def train_step(self) -> Dict[str, float]:
        """Performs one full training step, including sampling and network update."""
        if not self.buffer.is_ready(self.config.min_replay_size, self.config.min_replay_fill_pct):
            self.logger.log_text("Buffer not ready for training, skipping step.")
            return {}
        self.logger.log_text("Starting training step...")
        batch = self.buffer.sample(self.config.batch_size)
        observations, actions, targets = self._prepare_batch(batch)

        params = nnx.state(self.agent.network, nnx.Param)

        grad_fn = jax.value_and_grad(self._loss_fn, has_aux=True)
        (total_loss, losses), grads = grad_fn(params, observations, actions, targets)

        updates, self.agent.optimizer_state = self.agent.optimizer.update(grads, self.agent.optimizer_state, params)
        updated_params = optax.apply_updates(params, updates)

        nnx.update(self.agent.network, updated_params)

        losses_float = {f"train/{k}": float(v) for k, v in losses.items()}
        losses_float["total_loss"] = float(total_loss)
        self.logger.log_scalars(losses_float, self.train_step_count)
        self.train_step_count += 1
        return {"total_loss": float(total_loss), **losses_float}

    def _loss_fn(self, params: nnx.State, observations: chex.Array, actions: chex.Array, targets: Dict[str, chex.Array]) -> Tuple[chex.Array, Dict[str, chex.Array]]:
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
            _, _, value = self.agent.network.initial_inference(last_obs)
            n_step_return += (discount**n_steps) * float(jnp.asarray(value)[0, 0])

        return n_step_return

    def _compute_losses(self, network: CuMindNetwork, observations: chex.Array, actions: chex.Array, targets: Dict[str, chex.Array]) -> Dict[str, chex.Array]:
        """Computes the value, policy, and reward losses."""
        hidden_states, policy_logits, values = network.initial_inference(observations)

        values_squeezed = jnp.asarray(values).squeeze()
        target_values = jnp.asarray(targets["values"])
        value_loss = jnp.mean((values_squeezed - target_values) ** 2)

        policy_log_probs = jax.nn.log_softmax(policy_logits, axis=-1)
        target_policies = jnp.asarray(targets["policies"])
        policy_loss = -jnp.mean(jnp.sum(target_policies * policy_log_probs, axis=-1))

        reward_loss = jnp.array(0.0)
        current_states = hidden_states

        for step in range(self.config.num_unroll_steps):
            step_actions = jnp.asarray(actions[:, step])
            next_states, pred_rewards, _, _ = network.recurrent_inference(current_states, step_actions)

            pred_rewards_squeezed = jnp.asarray(pred_rewards).squeeze()
            target_rewards = jnp.asarray(targets["rewards"])[:, step]
            reward_loss += jnp.mean((pred_rewards_squeezed - target_rewards) ** 2)
            current_states = next_states

        if self.config.num_unroll_steps > 0:
            reward_loss /= self.config.num_unroll_steps

        return {"value_loss": value_loss, "policy_loss": policy_loss, "reward_loss": reward_loss}

    def save_checkpoint(self, path: str) -> None:
        """Saves the agent's state to a checkpoint file."""
        state = self.agent.save_state()
        save_checkpoint(state, path)

    def load_checkpoint(self, path: str) -> None:
        """Loads the agent's state from a checkpoint file."""
        state = load_checkpoint(path)
        self.agent.load_state(state)
