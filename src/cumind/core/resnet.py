"""ResNet architecture for reinforcement learning."""

from typing import Optional, Tuple, Union

import chex
import jax.numpy as jnp
from flax import nnx

from cumind.core.encoder import BaseEncoder, ConvEncoder, VectorEncoder


class ResNet(nnx.Module):
    """ResNet backbone supporting both vector and image inputs."""

    def __init__(self, input_dim: Union[int, Tuple[int, int, int]], hidden_dim: int, num_hidden_layers: int, rngs: nnx.Rngs, conv_channels: Optional[int] = None):
        """
        Initializes the ResNet model.

        Args:
            input_dim: Input dimension - int for vector input, (H, W, C) for image input
            hidden_dim: Dimension of hidden layers and output features
            num_hidden_layers: Number of residual blocks/layers
            conv_channels: Number of channels for convolutional layers (image input only)
            rngs: Random number generators for parameter initialization
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.conv_channels = conv_channels
        self.encoder: BaseEncoder
        # Handle 1D tuples as integers (e.g., (4,) -> 4)
        if isinstance(input_dim, tuple) and len(input_dim) == 1:
            input_dim: int = input_dim[0]

        if isinstance(input_dim, int):
            # Vector input: (input_dim,) -> hidden_dim
            observation_shape = (input_dim,)
            self.encoder = VectorEncoder(observation_shape=observation_shape, hidden_dim=hidden_dim, num_blocks=num_hidden_layers, rngs=rngs)
        elif isinstance(input_dim, tuple) and len(input_dim) == 3:
            # Image input: (height, width, channels) -> hidden_dim
            observation_shape = input_dim
            assert conv_channels is not None, "conv_channels must be provided for image input"
            self.encoder = ConvEncoder(observation_shape=observation_shape, hidden_dim=hidden_dim, num_blocks=num_hidden_layers, conv_channels=conv_channels, rngs=rngs)
        else:
            raise ValueError(f"Unsupported input_dim: {input_dim}. Use int or (X,) for vector input or (H, W, C) tuple for image input.")

    def __call__(self, x: chex.Array) -> chex.Array:
        """
        Forward pass through the ResNet.

        Args:
            x: Input tensor of shape (..., input_dim) for vector input
               or (..., H, W, C) for image input

        Returns:
            Output tensor of shape (..., hidden_dim)
        """
        x = jnp.asarray(x, dtype=jnp.float32)
        return self.encoder(x)
