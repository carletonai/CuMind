"""Utility functions and neural network modules for CuMind core components."""

import chex
import jax.numpy as jnp
from flax import nnx

from cumind.utils.logger import log


class ResidualBlock(nnx.Module):
    """A standard residual block with two convolutional layers."""

    def __init__(self, channels: int, rngs: nnx.Rngs):
        """Initializes the residual block.

        Args:
            channels: The number of input and output channels.
            rngs: Random number generators for layer initialization.
        """
        log.debug(f"Initializing ResidualBlock with {channels} channels.")
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
