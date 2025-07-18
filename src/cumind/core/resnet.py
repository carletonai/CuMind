"""ResNet architecture for CuMind."""

from typing import Tuple

import chex
from flax import nnx

from .encoder import BaseEncoder, ConvEncoder, VectorEncoder


class ResNet(nnx.Module):
    """
    General-purpose ResNet backbone supporting both vector and image inputs.

    Args:
        input_shape: Shape of the input data (tuple).
        hidden_dim: Dimension of hidden layers.
        num_blocks: Number of residual blocks.
        conv_channels: Number of channels for convolutional layers (used for image input).
        rngs: Random number generators for parameter initialization.

    Raises:
        ValueError: If input_shape is not 1D (vector) or 3D (image).
    """

    def __init__(self, hidden_dim: int, input_shape: Tuple[int, ...], num_blocks: int, conv_channels: int, rngs: nnx.Rngs):
        self.input_shape = input_shape
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.conv_channels = conv_channels
        self.encoder: BaseEncoder
        if len(input_shape) == 1:
            self.encoder = VectorEncoder(input_shape, hidden_dim, num_blocks, rngs)
        elif len(input_shape) == 3:
            self.encoder = ConvEncoder(input_shape, hidden_dim, num_blocks, conv_channels, rngs)
        else:
            raise ValueError("Unsupported observation shape")

    def __call__(self, x: chex.Array) -> chex.Array:
        return self.encoder(x)
