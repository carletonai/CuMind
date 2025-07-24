"""from typing import Tuple"""

from typing import Tuple

import chex
import jax.numpy as jnp
from flax import nnx

from cumind.core.blocks import ResidualBlock
from cumind.utils.logger import log


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
        log.debug(f"Initializing VectorEncoder with input dim {observation_shape[0]}, hidden dim {hidden_dim}, and {num_blocks} blocks.")

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

    def __init__(self, observation_shape: Tuple[int, ...], hidden_dim: int, num_blocks: int, conv_channels: int, rngs: nnx.Rngs):
        """Initializes the convolutional encoder.

        Args:
            observation_shape: The shape of the 3D observation (height, width, channels).
            hidden_dim: The output hidden dimension.
            num_blocks: The number of residual blocks.
            conv_channels: The number of channels for the convolutional layers.
            rngs: Random number generators for layer initialization.
        """
        super().__init__(observation_shape, hidden_dim, num_blocks, rngs)
        log.debug(f"Initializing ConvEncoder with input shape {observation_shape}, hidden dim {hidden_dim}, and {num_blocks} blocks.")

        self.initial_conv = nnx.Conv(observation_shape[-1], conv_channels, kernel_size=(3, 3), strides=(2, 2), padding=1, rngs=rngs)
        self.residual_blocks = [ResidualBlock(conv_channels, rngs) for _ in range(num_blocks)]
        self.final_dense = nnx.Linear(conv_channels, hidden_dim, rngs=rngs)

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
