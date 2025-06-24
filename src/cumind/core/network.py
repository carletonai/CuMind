"""Unified CuMind neural network components."""

from typing import Optional, Tuple

import chex
import jax
import jax.numpy as jnp
from flax import nnx


class ResidualBlock(nnx.Module):
    """Residual block for the neural network."""

    def __init__(self, channels: int, rngs: nnx.Rngs):
        """Initialize residual block layers.

        Args:
            channels: Number of input and output channels
            rngs: Random number generators for initialization
        """
        self.conv1 = nnx.Conv(channels, channels, kernel_size=(3, 3), padding=1, rngs=rngs)
        self.bn1 = nnx.BatchNorm(channels, rngs=rngs)
        self.conv2 = nnx.Conv(channels, channels, kernel_size=(3, 3), padding=1, rngs=rngs)
        self.bn2 = nnx.BatchNorm(channels, rngs=rngs)

    def __call__(self, x: chex.Array) -> chex.Array:
        """Forward pass with residual connection.

        Args:
            x: Input tensor of shape (batch, height, width, channels)

        Returns:
            Output tensor with same shape as input
        """
        residual = x
        x_array = jnp.asarray(x, dtype=jnp.float32)
        x_array = nnx.relu(self.bn1(self.conv1(x_array)))
        x_array = self.bn2(self.conv2(x_array))
        return nnx.relu(x_array + residual)


class BaseEncoder(nnx.Module):
    """Base class for observation encoders."""

    def __init__(self, observation_shape: Tuple[int, ...], hidden_dim: int, num_blocks: int, rngs: nnx.Rngs):
        """Initialize base encoder with configuration.

        Args:
            observation_shape: Shape of input observations
            hidden_dim: Dimension of hidden representation
            num_blocks: Number of processing blocks
            rngs: Random number generators for initialization
        """
        self.observation_shape = observation_shape
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        # Store rngs for potential use in subclasses
        self.rngs = rngs

    def __call__(self, observation: chex.Array) -> chex.Array:
        raise NotImplementedError("BaseEncoder is abstract")


class VectorEncoder(BaseEncoder):
    """Encoder for 1D vector observations."""

    def __init__(self, observation_shape: Tuple[int, ...], hidden_dim: int, num_blocks: int, rngs: nnx.Rngs):
        """Initialize vector encoder with fully connected layers.

        Args:
            observation_shape: Shape of 1D observation (e.g., (4,))
            hidden_dim: Output hidden dimension
            num_blocks: Number of fully connected blocks
            rngs: Random number generators for initialization
        """
        super().__init__(observation_shape, hidden_dim, num_blocks, rngs)

        input_dim = observation_shape[0]
        self.layers = []

        # Create fully connected layers
        current_dim = input_dim
        for _ in range(num_blocks):
            self.layers.append(nnx.Linear(current_dim, hidden_dim, rngs=rngs))
            current_dim = hidden_dim

    def __call__(self, observation: chex.Array) -> chex.Array:
        """Encode 1D observation through fully connected layers.

        Args:
            observation: Flattened observation of shape (batch, obs_dim)

        Returns:
            Hidden state tensor of shape (batch, hidden_dim)
        """
        x_array = jnp.asarray(observation, dtype=jnp.float32)
        for i, layer in enumerate(self.layers):
            x_array = layer(x_array)
            if i < len(self.layers) - 1:  # No ReLU after final layer
                x_array = nnx.relu(x_array)
        return x_array


class ConvEncoder(BaseEncoder):
    """Encoder for 3D image observations (e.g., Atari)."""

    def __init__(self, observation_shape: Tuple[int, ...], hidden_dim: int, num_blocks: int, rngs: nnx.Rngs):
        """Initialize convolutional encoder for image observations.

        Args:
            observation_shape: Shape of 3D observation (channels, height, width)
            hidden_dim: Output hidden dimension
            num_blocks: Number of residual blocks
            rngs: Random number generators for initialization
        """
        super().__init__(observation_shape, hidden_dim, num_blocks, rngs)

        # Initial conv to reduce spatial dimensions and increase channels
        self.initial_conv = nnx.Conv(observation_shape[-1], 32, kernel_size=(3, 3), strides=(2, 2), padding=1, rngs=rngs)

        # Residual blocks for feature processing
        self.residual_blocks = []
        for _ in range(num_blocks):
            self.residual_blocks.append(ResidualBlock(32, rngs))

        # Final dense layer to output hidden_dim
        self.final_dense = nnx.Linear(32, hidden_dim, rngs=rngs)

    def __call__(self, observation: chex.Array) -> chex.Array:
        """Encode 3D observation through convolutional layers.

        Args:
            observation: Image observation of shape (batch, height, width, channels)

        Returns:
            Hidden state tensor of shape (batch, hidden_dim)
        """
        x_array = jnp.asarray(observation, dtype=jnp.float32)
        x_array = nnx.relu(self.initial_conv(x_array))

        # Process through residual blocks
        for block in self.residual_blocks:
            x_array = jnp.asarray(block(x_array), dtype=jnp.float32)

        # Global average pooling to reduce spatial dimensions
        pooled = jnp.mean(x_array, axis=(1, 2))  # Average over height and width

        # Final dense layer
        return self.final_dense(pooled)


class RepresentationNetwork(nnx.Module):
    """rθ: Encode raw observations to hidden states."""

    def __init__(self, observation_shape: Tuple[int, ...], hidden_dim: int = 64, num_blocks: int = 4, rngs: Optional[nnx.Rngs] = None):
        """Initialize representation network with appropriate encoder.

        Args:
            observation_shape: Shape of input observations
            hidden_dim: Dimension of hidden representation
            num_blocks: Number of processing blocks
            rngs: Random number generators for initialization
        """
        if rngs is None:
            rngs = nnx.Rngs(params=jax.random.PRNGKey(0))

        self.observation_shape = observation_shape
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.encoder = self._create_encoder(rngs)

    def _create_encoder(self, rngs: nnx.Rngs) -> BaseEncoder:
        """Factory method to create appropriate encoder based on observation shape.

        Args:
            rngs: Random number generators for initialization

        Returns:
            VectorEncoder for 1D observations, ConvEncoder for 3D observations
        """
        if len(self.observation_shape) == 1:
            return VectorEncoder(self.observation_shape, self.hidden_dim, self.num_blocks, rngs)
        elif len(self.observation_shape) == 3:
            return ConvEncoder(self.observation_shape, self.hidden_dim, self.num_blocks, rngs)
        else:
            raise ValueError(f"Unsupported observation shape: {self.observation_shape}. Expected 1D vector or 3D image, got {len(self.observation_shape)}D")

    def __call__(self, observation: chex.Array) -> chex.Array:
        """Encode observation to hidden state.

        Args:
            observation: Input observation tensor

        Returns:
            Hidden state tensor of shape (batch, hidden_dim)
        """
        return self.encoder(observation)


class DynamicsNetwork(nnx.Module):
    """gθ: Predict next hidden state and reward from current state and action."""

    def __init__(self, hidden_dim: int, action_space_size: int, num_blocks: int = 4, rngs: Optional[nnx.Rngs] = None):
        """Initialize dynamics network with action embedding and processing blocks.

        Args:
            hidden_dim: Dimension of hidden states
            action_space_size: Number of possible actions
            num_blocks: Number of processing blocks
            rngs: Random number generators for initialization
        """
        if rngs is None:
            rngs = nnx.Rngs(params=jax.random.PRNGKey(0))

        self.hidden_dim = hidden_dim
        self.action_space_size = action_space_size

        # Action embedding to convert discrete actions to vectors
        self.action_embedding = nnx.Embed(action_space_size, hidden_dim, rngs=rngs)

        # Processing layers for state + action
        self.layers = []
        input_dim = hidden_dim * 2  # concatenated state and action
        for _ in range(num_blocks):
            self.layers.append(nnx.Linear(input_dim, hidden_dim, rngs=rngs))
            input_dim = hidden_dim

        # Reward prediction head
        self.reward_head = nnx.Linear(hidden_dim, 1, rngs=rngs)

    def __call__(self, state: chex.Array, action: chex.Array) -> Tuple[chex.Array, chex.Array]:
        """Predict next state and reward.

        Args:
            state: Current hidden state of shape (batch, hidden_dim)
            action: Action indices of shape (batch,)

        Returns:
            Tuple of (next_state, reward) tensors
        """
        # Embed actions and concatenate with state
        state_array = jnp.asarray(state, dtype=jnp.float32)
        action_array = jnp.asarray(action, dtype=jnp.int32)
        action_embedded = self.action_embedding(action_array)
        x = jnp.concatenate([state_array, action_embedded], axis=-1)

        # Process through layers with residual connections
        for i, layer in enumerate(self.layers):
            residual = x if i > 0 and x.shape[-1] == self.hidden_dim else None
            x = layer(x)
            if residual is not None:
                x = x + residual  # Skip connection for same-sized tensors
            x = nnx.relu(x)

        # Predict reward
        reward = self.reward_head(x)

        # Next state is the processed hidden representation
        next_state = x

        return next_state, reward


class PredictionNetwork(nnx.Module):
    """fθ: Predict policy and value from hidden state."""

    def __init__(self, hidden_dim: int, action_space_size: int, rngs: Optional[nnx.Rngs] = None):
        """Initialize prediction network with policy and value heads.

        Args:
            hidden_dim: Dimension of input hidden states
            action_space_size: Number of possible actions for policy head
            rngs: Random number generators for initialization
        """
        if rngs is None:
            rngs = nnx.Rngs(params=jax.random.PRNGKey(0))

        # Policy head: outputs action logits
        self.policy_head = nnx.Linear(hidden_dim, action_space_size, rngs=rngs)

        # Value head: outputs single scalar value
        self.value_head = nnx.Linear(hidden_dim, 1, rngs=rngs)

    def __call__(self, hidden_state: chex.Array) -> Tuple[chex.Array, chex.Array]:
        """Predict policy logits and value from hidden state.

        Args:
            hidden_state: Hidden state tensor of shape (batch, hidden_dim)

        Returns:
            Tuple of (policy_logits, value) where:
            - policy_logits: shape (batch, action_space_size)
            - value: shape (batch, 1)
        """
        hidden_state_array = jnp.asarray(hidden_state, dtype=jnp.float32)
        policy_logits = self.policy_head(hidden_state_array)
        value = self.value_head(hidden_state_array)
        return policy_logits, value


class CuMindNetwork(nnx.Module):
    """Complete CuMind neural network combining all three networks."""

    def __init__(self, observation_shape: Tuple[int, ...], action_space_size: int, hidden_dim: int = 64, num_blocks: int = 4, rngs: Optional[nnx.Rngs] = None):
        """Initialize complete CuMind network.

        Args:
            observation_shape: Shape of input observations
            action_space_size: Number of possible actions
            hidden_dim: Dimension of hidden representations
            num_blocks: Number of processing blocks
            rngs: Random number generators for initialization
        """
        if rngs is None:
            rngs = nnx.Rngs(params=jax.random.PRNGKey(0))

        # Create the three networks
        self.representation_network = RepresentationNetwork(observation_shape, hidden_dim, num_blocks, rngs)
        self.dynamics_network = DynamicsNetwork(hidden_dim, action_space_size, num_blocks, rngs)
        self.prediction_network = PredictionNetwork(hidden_dim, action_space_size, rngs)

    def initial_inference(self, observation: chex.Array) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """Initial step: observation -> hidden state -> policy, value.

        Args:
            observation: Input observation tensor

        Returns:
            Tuple of (hidden_state, policy_logits, value)
        """
        hidden_state = self.representation_network(observation)
        policy_logits, value = self.prediction_network(hidden_state)
        return hidden_state, policy_logits, value

    def recurrent_inference(self, hidden_state: chex.Array, action: chex.Array) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
        """Recurrent step: hidden state + action -> next hidden state, reward, policy, value.

        Args:
            hidden_state: Current hidden state
            action: Action to take

        Returns:
            Tuple of (next_hidden_state, reward, policy_logits, value)
        """
        next_hidden_state, reward = self.dynamics_network(hidden_state, action)
        policy_logits, value = self.prediction_network(next_hidden_state)
        return next_hidden_state, reward, policy_logits, value
