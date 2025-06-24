"""CuMind agent implementation."""

from typing import Any, Dict, List, Tuple, cast

import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax  # type: ignore
from flax import nnx

from ..config import Config
from ..core.mcts import MCTS
from ..core.network import CuMindNetwork


class Agent:
    """CuMind agent for training and inference."""

    def __init__(self, config: Config):
        """Initialize CuMind agent with network, optimizer, and MCTS.

        Args:
            config: Config with network architecture and training parameters
        """
        self.config = config

        # Create network with random initialization
        key = jax.random.PRNGKey(0)
        rngs = nnx.Rngs(params=key)

        self.network = CuMindNetwork(observation_shape=config.observation_shape, action_space_size=config.action_space_size, hidden_dim=config.hidden_dim, num_blocks=config.num_blocks, rngs=rngs)

        # Setup optimizer (fix weight_decay parameter)
        self.optimizer = optax.adamw(learning_rate=config.learning_rate, weight_decay=config.weight_decay)
        self.optimizer_state = self.optimizer.init(nnx.state(self.network, nnx.Param))

        # Initialize MCTS
        self.mcts = MCTS(config)

    def select_action(self, observation: np.ndarray, training: bool = True) -> int:
        """Select action using MCTS search from current observation.

        Args:
            observation: Current game state observation
            training: If True, sample from action probabilities; if False, take best action

        Returns:
            Selected action index
        """
        # Convert observation to tensor and get initial hidden state
        obs_tensor = jnp.array(observation)[None]  # Add batch dimension
        hidden_state, _, _ = self.network.initial_inference(obs_tensor)
        hidden_state_array = jnp.asarray(hidden_state, dtype=jnp.float32)[0]  # Remove batch dimension

        # Use MCTS to get action probabilities
        action_probs = self.mcts.search(self.network, hidden_state_array, add_noise=training)

        if training:
            # Sample action from probabilities
            action_idx = np.random.choice(len(action_probs), p=action_probs)
            return int(action_idx)
        else:
            # Take best action
            action_idx = int(np.argmax(action_probs))
            return action_idx

    def train_step(self, batch: List[Any]) -> Dict[str, float]:
        """Perform one training step on a batch of replay data.

        Args:
            batch: List of game trajectories from replay buffer

        Returns:
            Dictionary of loss values for monitoring
        """
        # Prepare batch data
        observations, actions, targets = self._prepare_batch(batch)

        # Get current parameters
        params = nnx.state(self.network, nnx.Param)

        # Define loss function
        def loss_fn(p: Any) -> Tuple[chex.Array, Dict[str, chex.Array]]:
            # Create a temporary network with these parameters
            temp_network = nnx.clone(self.network)
            nnx.update(temp_network, {"params": p})
            losses = self._compute_losses(temp_network, observations, actions, targets)
            total_loss = jnp.sum(jnp.array(list(losses.values())))
            return total_loss, losses

        # Compute gradients
        grad_fn = jax.value_and_grad(lambda p: loss_fn(p)[0], has_aux=True)
        (_, losses), grads = grad_fn(params)

        # Apply gradients
        updates, self.optimizer_state = self.optimizer.update(grads, self.optimizer_state, params) # type: ignore
        updated_params = optax.apply_updates(params, updates) # type: ignore

        # Update network with new parameters
        nnx.update(self.network, {"params": updated_params})

        # Convert to float for logging
        return {k: float(v) for k, v in losses.items()}

    def _prepare_batch(self, batch: List[Any]) -> Tuple[chex.Array, chex.Array, Dict[str, chex.Array]]:
        """Extract and format training data from replay buffer batch.

        Args:
            batch: Raw batch data from replay buffer

        Returns:
            Tuple of (observations, actions, targets) tensors ready for training
        """
        # Extract data from batch trajectories
        observations: List[Any] = []
        action_sequences: List[List[int]] = []
        value_targets: List[float] = []
        reward_targets: List[List[float]] = []
        policy_targets: List[List[float]] = []

        for trajectory in batch:
            # Each trajectory is a list of steps
            if len(trajectory) == 0:
                continue

            # Use first observation
            obs = trajectory[0]["observation"]
            observations.append(obs)

            # Extract action sequence (limited by unroll steps)
            actions = [step["action"] for step in trajectory[: self.config.num_unroll_steps]]
            # Pad if needed
            while len(actions) < self.config.num_unroll_steps:
                actions.append(0)  # Pad with action 0
            action_sequences.append(actions)

            # Extract rewards for unroll steps
            rewards = [step["reward"] for step in trajectory[: self.config.num_unroll_steps]]
            while len(rewards) < self.config.num_unroll_steps:
                rewards.append(0.0)
            reward_targets.append(rewards)  # Append the list

            # Simple value target: discounted sum of future rewards
            total_reward = sum(step["reward"] for step in trajectory)
            value_targets.append(total_reward)  # Append the scalar value

            # Simple policy target: uniform distribution (would use MCTS probs in full implementation)
            uniform_policy = [1.0 / self.config.action_space_size] * self.config.action_space_size
            policy_targets.append(uniform_policy)

        # Convert to JAX arrays
        observations_array = jnp.array(observations)
        actions_array = jnp.array(action_sequences)

        targets = {"values": jnp.array(value_targets), "rewards": jnp.array(reward_targets), "policies": jnp.array(policy_targets)}

        return cast(chex.Array, observations_array), cast(chex.Array, actions_array), cast(Dict[str, chex.Array], targets)

    def _compute_losses(
        self,
        network: CuMindNetwork,
        observations: chex.Array,
        actions: chex.Array,
        targets: Dict[str, chex.Array],
    ) -> Dict[str, chex.Array]:
        """Compute CuMind losses for value, policy, and reward predictions.

        Args:
            network: CuMind network for prediction
            observations: Batch observations, shape (batch_size, *obs_shape)
            actions: Action sequences, shape (batch_size, num_unroll_steps)
            targets: Dict with "values", "policies", "rewards" target tensors

        Returns:
            Dict of loss tensors: {"value_loss", "policy_loss", "reward_loss"}
        """
        # Initial inference
        hidden_states, policy_logits, values = network.initial_inference(observations)

        # Initial step losses
        value_loss = jnp.mean((values - jnp.asarray(targets["values"])) ** 2)
        policy_probs_log = jax.nn.log_softmax(policy_logits, axis=-1)
        policy_sum = jnp.sum(jnp.asarray(targets["policies"]) * jnp.asarray(policy_probs_log), axis=-1)
        policy_loss = jnp.negative(jnp.mean(policy_sum))

        reward_loss = jnp.array(0.0)

        # Unroll dynamics for each step
        current_states = hidden_states
        for step in range(self.config.num_unroll_steps):
            actions_array = jnp.asarray(actions, dtype=jnp.int32)
            if step >= actions_array.shape[1]:
                break

            # Get actions for this step
            step_actions = actions_array[:, step]

            # Recurrent inference
            next_states, predicted_rewards, _, _ = network.recurrent_inference(current_states, step_actions)

            # Reward loss for this step
            rewards_array = jnp.asarray(targets["rewards"], dtype=jnp.float32)
            if step < rewards_array.shape[1]:
                target_rewards = rewards_array[:, step : step + 1]
                reward_loss += jnp.mean((predicted_rewards - target_rewards) ** 2)

            current_states = next_states

        # Average reward loss over unroll steps
        if self.config.num_unroll_steps > 0:
            reward_loss = reward_loss / float(self.config.num_unroll_steps)

        return {"value_loss": value_loss, "policy_loss": policy_loss, "reward_loss": reward_loss}

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint with network weights, optimizer state, and config.

        Args:
            path: File path to save checkpoint
        """
        import pickle
        from pathlib import Path

        checkpoint_data = {"network_state": nnx.state(self.network), "optimizer_state": self.optimizer_state, "config": self.config}

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(checkpoint_data, f)

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint to restore training state.

        Args:
            path: File path to load checkpoint from
        """
        import pickle

        with open(path, "rb") as f:
            checkpoint_data = pickle.load(f)

        # Restore network state
        nnx.update(self.network, checkpoint_data["network_state"])

        # Restore optimizer state
        self.optimizer_state = checkpoint_data["optimizer_state"]
