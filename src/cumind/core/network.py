"""Unified CuMind neural network components."""

from typing import Optional, Tuple

import chex
import jax
import jax.numpy as jnp
from flax import nnx


class ResidualBlock(nnx.Module):
    """A standard residual block with two convolutional layers."""

    def __init__(self, channels: int, rngs: nnx.Rngs):
        """Initializes the residual block.

        Args:
            channels: The number of input and output channels.
            rngs: Random number generators for layer initialization.
        """
        self.conv1 = nnx.Conv(channels, channels, kernel_size=(3, 3), padding=1, rngs=rngs)
        self.bn1 = nnx.BatchNorm(channels, use_running_average=True, rngs=rngs)
        self.conv2 = nnx.Conv(channels, channels, kernel_size=(3, 3), padding=1, rngs=rngs)
        self.bn2 = nnx.BatchNorm(channels, use_running_average=True, rngs=rngs)

    def __call__(self, x: chex.Array) -> chex.Array:
        """Performs a forward pass through the residual block.

        Args:
            x: The input tensor of shape (batch, height, width, channels).

        Returns:
            The output tensor with the same shape as the input.
        """
        residual = x
        x_array = jnp.asarray(x, dtype=jnp.float32)
        x_array = self.conv1(x_array)
        x_array = self.bn1(x_array)
        x_array = nnx.relu(x_array)
        x_array = self.conv2(x_array)
        x_array = self.bn2(x_array)
        return nnx.relu(x_array + residual)


class BaseEncoder(nnx.Module):
    """Abstract base class for observation encoders."""

    def __init__(self, observation_shape: Tuple[int, ...], hidden_dim: int, num_blocks: int, rngs: nnx.Rngs):
        """Initializes the base encoder.

        Args:
            observation_shape: The shape of the input observations.
            hidden_dim: The dimension of the output hidden representation.
            num_blocks: The number of processing blocks (e.g., residual or dense).
            rngs: Random number generators for layer initialization.
        """
        self.observation_shape = observation_shape
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.rngs = rngs

    def __call__(self, observation: chex.Array) -> chex.Array:
        """Encodes an observation into a hidden state."""
        raise NotImplementedError("BaseEncoder is an abstract class.")


class VectorEncoder(BaseEncoder):
    """An encoder for 1D vector observations using fully connected layers."""

    def __init__(self, observation_shape: Tuple[int, ...], hidden_dim: int, num_blocks: int, rngs: nnx.Rngs):
        """Initializes the vector encoder.

        Args:
            observation_shape: The shape of the 1D observation (e.g., (4,)).
            hidden_dim: The output hidden dimension.
            num_blocks: The number of fully connected blocks.
            rngs: Random number generators for layer initialization.
        """
        super().__init__(observation_shape, hidden_dim, num_blocks, rngs)

        input_dim = observation_shape[0]
        self.layers = []

        current_dim = input_dim
        for i in range(num_blocks):
            output_dim = hidden_dim if i == num_blocks - 1 else hidden_dim
            self.layers.append(nnx.Linear(current_dim, output_dim, rngs=rngs))
            current_dim = output_dim

    def __call__(self, observation: chex.Array) -> chex.Array:
        """Encodes a 1D observation into a hidden state.

        Args:
            observation: A flattened observation of shape (batch, obs_dim).

        Returns:
            A hidden state tensor of shape (batch, hidden_dim).
        """
        x = jnp.asarray(observation, dtype=jnp.float32)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = nnx.relu(x)
        return x


class ConvEncoder(BaseEncoder):
    """An encoder for 3D image observations using convolutional layers."""

    def __init__(self, observation_shape: Tuple[int, ...], hidden_dim: int, num_blocks: int, rngs: nnx.Rngs):
        """Initializes the convolutional encoder.

        Args:
            observation_shape: The shape of the 3D observation (height, width, channels).
            hidden_dim: The output hidden dimension.
            num_blocks: The number of residual blocks.
            rngs: Random number generators for layer initialization.
        """
        super().__init__(observation_shape, hidden_dim, num_blocks, rngs)

        self.initial_conv = nnx.Conv(observation_shape[-1], 32, kernel_size=(3, 3), strides=(2, 2), padding=1, rngs=rngs)
        self.residual_blocks = [ResidualBlock(32, rngs) for _ in range(num_blocks)]
        self.final_dense = nnx.Linear(32, hidden_dim, rngs=rngs)

    def __call__(self, observation: chex.Array) -> chex.Array:
        """Encodes a 3D observation into a hidden state.

        Args:
            observation: An image observation of shape (batch, height, width, channels).

        Returns:
            A hidden state tensor of shape (batch, hidden_dim).
        """
        x = jnp.asarray(observation, dtype=jnp.float32)
        x = nnx.relu(self.initial_conv(x))

        for block in self.residual_blocks:
            x = jnp.asarray(block(x), dtype=jnp.float32)

        pooled = jnp.mean(x, axis=(1, 2))
        return self.final_dense(pooled)


class RepresentationNetwork(nnx.Module):
    """The representation network (r_theta) that encodes raw observations into hidden states."""

    def __init__(self, observation_shape: Tuple[int, ...], hidden_dim: int = 64, num_blocks: int = 4, rngs: Optional[nnx.Rngs] = None):
        """Initializes the representation network.

        Args:
            observation_shape: The shape of the input observations.
            hidden_dim: The dimension of the hidden representation.
            num_blocks: The number of processing blocks in the encoder.
            rngs: Random number generators for layer initialization.
        """
        if rngs is None:
            rngs = nnx.Rngs(params=jax.random.PRNGKey(0))

        self.observation_shape = observation_shape
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.encoder = self._create_encoder(rngs)

    def _create_encoder(self, rngs: nnx.Rngs) -> BaseEncoder:
        """Creates the appropriate encoder based on the observation shape."""
        if len(self.observation_shape) == 1:
            return VectorEncoder(self.observation_shape, self.hidden_dim, self.num_blocks, rngs)
        if len(self.observation_shape) == 3:
            return ConvEncoder(self.observation_shape, self.hidden_dim, self.num_blocks, rngs)
        raise ValueError(f"Unsupported observation shape: {self.observation_shape}")

    def __call__(self, observation: chex.Array) -> chex.Array:
        """Encodes an observation into a hidden state.

        Args:
            observation: The input observation tensor.

        Returns:
            A hidden state tensor of shape (batch, hidden_dim).
        """
        return self.encoder(observation)


class DynamicsNetwork(nnx.Module):
    """The dynamics network (g_theta) that predicts the next hidden state and reward."""

    def __init__(self, hidden_dim: int, action_space_size: int, num_blocks: int = 4, rngs: Optional[nnx.Rngs] = None):
        """Initializes the dynamics network.

        Args:
            hidden_dim: The dimension of the hidden states.
            action_space_size: The number of possible actions.
            num_blocks: The number of processing blocks.
            rngs: Random number generators for layer initialization.
        """
        if rngs is None:
            rngs = nnx.Rngs(params=jax.random.PRNGKey(0))

        self.hidden_dim = hidden_dim
        self.action_space_size = action_space_size
        self.action_embedding = nnx.Embed(action_space_size, hidden_dim, rngs=rngs)

        self.layers = []
        for _ in range(num_blocks):
            self.layers.append(nnx.Linear(hidden_dim, hidden_dim, rngs=rngs))

        self.reward_head = nnx.Linear(hidden_dim, 1, rngs=rngs)

    def __call__(self, state: chex.Array, action: chex.Array) -> Tuple[chex.Array, chex.Array]:
        """Predicts the next hidden state and reward.

        Args:
            state: The current hidden state of shape (batch, hidden_dim).
            action: The action indices of shape (batch,).

        Returns:
            A tuple containing the next hidden state and the predicted reward.
        """
        action_embedded = self.action_embedding(jnp.asarray(action, dtype=jnp.int32))
        x = jnp.asarray(state, dtype=jnp.float32) + action_embedded

        for layer in self.layers:
            residual = x
            x = nnx.relu(layer(x))
            x = x + residual

        reward = self.reward_head(x)
        return x, reward


class PredictionNetwork(nnx.Module):
    """The prediction network (f_theta) that predicts the policy and value from a hidden state."""

    def __init__(self, hidden_dim: int, action_space_size: int, rngs: Optional[nnx.Rngs] = None):
        """Initializes the prediction network.

        Args:
            hidden_dim: The dimension of the input hidden states.
            action_space_size: The number of possible actions for the policy head.
            rngs: Random number generators for layer initialization.
        """
        if rngs is None:
            rngs = nnx.Rngs(params=jax.random.PRNGKey(0))

        self.policy_head = nnx.Linear(hidden_dim, action_space_size, rngs=rngs)
        self.value_head = nnx.Linear(hidden_dim, 1, rngs=rngs)

    def __call__(self, hidden_state: chex.Array) -> Tuple[chex.Array, chex.Array]:
        """Predicts the policy logits and value from a hidden state.

        Args:
            hidden_state: A hidden state tensor of shape (batch, hidden_dim).

        Returns:
            A tuple of (policy_logits, value).
        """
        x = jnp.asarray(hidden_state, dtype=jnp.float32)
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        return policy_logits, value


class CuMindNetwork(nnx.Module):
    """The complete CuMind network, combining representation, dynamics, and prediction."""

    def __init__(self, observation_shape: Tuple[int, ...], action_space_size: int, hidden_dim: int = 64, num_blocks: int = 4, rngs: Optional[nnx.Rngs] = None):
        """Initializes the complete CuMind network.

        Args:
            observation_shape: The shape of the input observations.
            action_space_size: The number of possible actions.
            hidden_dim: The dimension of the hidden representations.
            num_blocks: The number of processing blocks in the networks.
            rngs: Random number generators for layer initialization.
        """
        if rngs is None:
            rngs = nnx.Rngs(params=jax.random.PRNGKey(0))

        self.representation_network = RepresentationNetwork(observation_shape, hidden_dim, num_blocks, rngs)
        self.dynamics_network = DynamicsNetwork(hidden_dim, action_space_size, num_blocks, rngs)
        self.prediction_network = PredictionNetwork(hidden_dim, action_space_size, rngs)

    def initial_inference(self, observation: chex.Array) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """Performs the initial inference step from an observation.

        Args:
            observation: The input observation tensor.

        Returns:
            A tuple of (hidden_state, policy_logits, value).
        """
        hidden_state = self.representation_network(observation)
        policy_logits, value = self.prediction_network(hidden_state)
        return hidden_state, policy_logits, value

    def recurrent_inference(self, hidden_state: chex.Array, action: chex.Array) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
        """Performs a recurrent inference step from a hidden state and action.

        Args:
            hidden_state: The current hidden state.
            action: The action to take.

        Returns:
            A tuple of (next_hidden_state, reward, policy_logits, value).
        """
        next_hidden_state, reward = self.dynamics_network(hidden_state, action)
        policy_logits, value = self.prediction_network(next_hidden_state)
        return next_hidden_state, reward, policy_logits, value
