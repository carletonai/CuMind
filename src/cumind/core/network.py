"""Unified CuMind neural network"""

from typing import Callable, Tuple

import chex
from flax import nnx

from ..utils.logger import log


class CuMindNetwork(nnx.Module):
    """The complete CuMind network, combining representation, dynamics, and prediction."""

    def __init__(self, representation_network: Callable[[chex.Array], chex.Array], dynamics_network: Callable[[chex.Array, chex.Array], Tuple[chex.Array, chex.Array]], prediction_network: Callable[[chex.Array], Tuple[chex.Array, chex.Array]], rngs: nnx.Rngs):
        """Initializes the complete CuMind network.

        Args:
            representation_network: The network responsible for encoding observations into latent representations.
            dynamics_network: The network that models environment dynamics in the latent space.
            prediction_network: The network that predicts policy and value from latent states.
            rngs: Random number generators for layer initialization.
        """
        log.info(f"Initializing CuMindNetwork with representation_network={type(representation_network).__name__}, dynamics_network={type(dynamics_network).__name__}, prediction_network={type(prediction_network).__name__}")
        self.representation_network = representation_network
        self.dynamics_network = dynamics_network
        self.prediction_network = prediction_network

    def initial_inference(self, observation: chex.Array) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """Performs the initial inference step from an observation.

        Args:
            observation: The input observation tensor.

        Returns:
            A tuple of (hidden_state, policy_logits, value).
        """
        log.debug(f"Initial inference with observation shape: {observation.shape}")
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
        log.debug(f"Recurrent inference with hidden state shape: {hidden_state.shape} and action shape: {action.shape}")
        next_hidden_state, reward = self.dynamics_network(hidden_state, action)
        policy_logits, value = self.prediction_network(next_hidden_state)
        return next_hidden_state, reward, policy_logits, value
