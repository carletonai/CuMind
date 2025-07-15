"""MLP neural network architectures."""

from typing import Tuple

import chex
import jax.numpy as jnp
from flax import nnx


class MLPWithEmbedding(nnx.Module):
    """MLP with embedding layer and residual connections."""

    def __init__(self, hidden_dim: int, embedding_size: int, num_blocks: int, rngs: nnx.Rngs):
        self.hidden_dim = hidden_dim
        self.embedding_size = embedding_size
        self.num_blocks = num_blocks
        self.embedding = nnx.Embed(embedding_size, hidden_dim, rngs=rngs)
        self.layers = [nnx.Linear(hidden_dim, hidden_dim, rngs=rngs) for _ in range(num_blocks)]
        self.output_head = nnx.Linear(hidden_dim, 1, rngs=rngs)

    def __call__(self, state: chex.Array, embedding_idx: chex.Array) -> Tuple[chex.Array, chex.Array]:
        """Forward pass through the network.

        Args:
            state: Input state tensor.
            embedding_idx: Embedding indices.

        Returns:
            Tuple of (transformed_state, output).
        """
        embedded = self.embedding(jnp.asarray(embedding_idx, dtype=jnp.int32))
        x = jnp.asarray(state, dtype=jnp.float32) + embedded

        for layer in self.layers:
            residual = x
            x = nnx.relu(layer(x))
            x = x + residual

        output = self.output_head(x)
        return x, output


class MLPDual(nnx.Module):
    """MLP with dual output heads."""

    def __init__(self, hidden_dim: int, output_size: int, rngs: nnx.Rngs):
        self.hidden_dim = hidden_dim
        self.output_size = output_size
        self.head1 = nnx.Linear(hidden_dim, output_size, rngs=rngs)
        self.head2 = nnx.Linear(hidden_dim, 1, rngs=rngs)

    def __call__(self, hidden_state: chex.Array) -> Tuple[chex.Array, chex.Array]:
        """Forward pass through the network.

        Args:
            hidden_state: Input hidden state tensor.

        Returns:
            Tuple of (output1, output2).
        """
        x = jnp.asarray(hidden_state, dtype=jnp.float32)
        output1 = self.head1(x)
        output2 = self.head2(x)
        return output1, output2
