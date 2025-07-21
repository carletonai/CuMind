"""MLP neural network architectures for reinforcement learning."""

from typing import Tuple

import chex
import jax.numpy as jnp
from flax import nnx


class MLPWithEmbedding(nnx.Module):
    """MLP with embedding layer and residual connections."""

    def __init__(self, input_dim: int, hidden_dim: int, num_hidden_layers: int, embedding_size: int, rngs: nnx.Rngs):
        """
        Initializes the MLP (Multi-Layer Perceptron) model.

        Args:
            input_dim: Dimension of input state vectors
            hidden_dim: Dimension of hidden layers
            num_hidden_layers: Number of hidden layers in the network
            embedding_size: Size of action embedding vocabulary
            rngs: Random number generators for parameter initialization
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.embedding_size = embedding_size
        self.action_embedding = nnx.Embed(embedding_size, hidden_dim, rngs=rngs)
        self.input_projection = nnx.Linear(input_dim, hidden_dim, rngs=rngs)
        self.hidden_layers = [nnx.Linear(hidden_dim, hidden_dim, rngs=rngs) for _ in range(num_hidden_layers)]
        self.output_layer = nnx.Linear(hidden_dim, 1, rngs=rngs)

    def __call__(self, state: chex.Array, embedding_idx: chex.Array) -> Tuple[chex.Array, chex.Array]:
        """
        Forward pass through the network.

        Args:
            state: Input state tensor of shape (..., input_dim)
            embedding_idx: Action embedding indices of shape (...,)

        Returns:
            Tuple of (hidden_features, output_value)
        """
        state = jnp.asarray(state, dtype=jnp.float32)
        embedding_idx = jnp.asarray(embedding_idx, dtype=jnp.int32)
        x = self.input_projection(state)

        action_emb = self.action_embedding(embedding_idx)
        x = x + action_emb

        for layer in self.hidden_layers:
            residual = x
            x = nnx.relu(layer(x))
            x = x + residual
        output = self.output_layer(x)

        return x, output


class MLPDual(nnx.Module):
    """MLP with dual output heads for policy and value."""

    def __init__(self, input_dim: int, hidden_dim: int, num_hidden_layers: int, num_actions: int, rngs: nnx.Rngs):
        """
        Initialize dual-head MLP for policy and value.

        Args:
            input_dim: Dimension of input state vectors
            hidden_dim: Dimension of hidden layers
            num_hidden_layers: Number of hidden layers
            num_actions: Number of possible actions (for policy head)
            rngs: Random number generators for parameter initialization
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_actions = num_actions
        self.input_projection = nnx.Linear(input_dim, hidden_dim, rngs=rngs)
        self.hidden_layers = [nnx.Linear(hidden_dim, hidden_dim, rngs=rngs) for _ in range(num_hidden_layers)]
        self.policy_head = nnx.Linear(hidden_dim, num_actions, rngs=rngs)
        self.value_head = nnx.Linear(hidden_dim, 1, rngs=rngs)

    def __call__(self, state: chex.Array) -> Tuple[chex.Array, chex.Array]:
        """
        Forward pass through the network.

        Args:
            state: Input state tensor of shape (..., input_dim)

        Returns:
            Tuple of (policy_logits, value)
        """
        x = jnp.asarray(state, dtype=jnp.float32)
        x = self.input_projection(x)
        for layer in self.hidden_layers:
            x = nnx.relu(layer(x))

        policy_logits = self.policy_head(x)
        value = self.value_head(x)

        return policy_logits, value
