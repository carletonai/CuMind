"""Unified CuMind neural network"""

from typing import Callable, Optional, Tuple

import chex
import jax
from flax import nnx

from cumind.utils.logger import log


class CuMindNetwork(nnx.Module):
    """The complete CuMind network, combining representation, dynamics, and prediction, with target prediction support."""

    def __init__(self, representation_network: Callable[[chex.Array], chex.Array], dynamics_network: Callable[[chex.Array, chex.Array], Tuple[chex.Array, chex.Array]], prediction_network: Callable[[chex.Array], Tuple[chex.Array, chex.Array]]) -> None:
        """Initializes the complete CuMind network.

        Args:
            representation_network: The network responsible for encoding observations into latent representations.
            dynamics_network: The network that models environment dynamics in the latent space.
            prediction_network: The network that predicts policy and value from latent states.
        """
        log.info(f"Initializing CuMindNetwork with representation_network={type(representation_network).__name__}, dynamics_network={type(dynamics_network).__name__}, prediction_network={type(prediction_network).__name__}")
        self.representation_network = representation_network
        self.dynamics_network = dynamics_network
        self.prediction_network = prediction_network
        # Target prediction network (initialized as a clone of the online prediction network)
        self.target_prediction_network = nnx.clone(prediction_network)

    def update_target_prediction_network(self, hard: bool = True, tau: Optional[float] = None) -> None:
        """Update the target prediction network. Hard copy by default, or soft update if tau is provided."""
        log.debug(f"Updating target prediction network: hard={hard}, tau={tau}")
        if hard or tau is None:
            self.target_prediction_network = nnx.clone(self.prediction_network)
            log.debug("Target prediction network updated with hard copy")
        else:
            # Polyak averaging: target = tau * online + (1-tau) * target
            online_params = nnx.state(self.prediction_network, nnx.Param)
            target_params = nnx.state(self.target_prediction_network, nnx.Param)
            new_params = jax.tree_util.tree_map(lambda o, t: tau * o + (1 - tau) * t, online_params, target_params)
            nnx.update(self.target_prediction_network, new_params)
            log.debug("Target prediction network updated with soft update (Polyak averaging)")

    def initial_inference(self, observation: chex.Array, use_target: bool = False) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """Performs the initial inference step from an observation.

        Args:
            observation: The input observation tensor.

        Returns:
            A tuple of (hidden_state, policy_logits, value).
        """
        log.debug(f"Initial inference with observation shape: {observation.shape}, use_target: {use_target}")
        hidden_state: chex.Array = self.representation_network(observation)
        if use_target:
            policy_logits, value = self.target_prediction_network(hidden_state)
        else:
            policy_logits, value = self.prediction_network(hidden_state)
        return hidden_state, policy_logits, value

    def recurrent_inference(self, hidden_state: chex.Array, action: chex.Array, use_target: bool = False) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
        """Performs a recurrent inference step from a hidden state and action.

        Args:
            hidden_state: The current hidden state.
            action: The action to take.

        Returns:
            A tuple of (next_hidden_state, reward, policy_logits, value).
        """
        log.debug(f"Recurrent inference with hidden state shape: {hidden_state.shape} and action shape: {action.shape}")
        next_hidden_state: chex.Array
        reward: chex.Array
        next_hidden_state, reward = self.dynamics_network(hidden_state, action)
        if use_target:
            policy_logits, value = self.target_prediction_network(next_hidden_state)
        else:
            policy_logits, value = self.prediction_network(next_hidden_state)
        return next_hidden_state, reward, policy_logits, value
