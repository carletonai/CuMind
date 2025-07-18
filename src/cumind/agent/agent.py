"""CuMind agent implementation."""

from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax  # type: ignore
from flax import nnx

from cumind.core.mcts import MCTS
from cumind.core.network import CuMindNetwork
from cumind.utils.config import cfg
from cumind.utils.logger import log
from cumind.utils.prng import key


class Agent:
    """CuMind agent for training and inference."""

    def __init__(self, existing_state: Optional[Dict[str, Any]] = None):
        """Initialize CuMind agent with network, optimizer, and MCTS.

        Args:
            existing_state: Optional dictionary to load agent state from.
        """
        log.info("Initializing CuMind agent.")

        self.device = jax.devices(cfg.device)[0]
        log.info(f"Using device: {self.device}")

        log.info(f"Creating CuMindNetwork with observation shape {cfg.env.observation_shape} and action space size {cfg.env.action_space_size}")

        with jax.default_device(self.device):
            self.network = CuMindNetwork(representation_network=cfg.representation(), dynamics_network=cfg.dynamics(), prediction_network=cfg.prediction())

            log.info("Creating target network.")
            self.target_network = nnx.clone(self.network)

            log.info(f"Setting up AdamW optimizer with learning rate {cfg.training.learning_rate} and weight decay {cfg.training.weight_decay}")
            self.optimizer = optax.adamw(learning_rate=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)

            if existing_state:
                log.info("Loading agent state from existing state.")
                self.load_state(existing_state)
            else:
                log.info("Initializing new optimizer state.")
                self.optimizer_state = self.optimizer.init(nnx.state(self.network, nnx.Param))

        self.mcts = MCTS(self.network)
        log.info("Agent initialization complete.")

    def select_action(self, observation: np.ndarray, training: bool = False) -> Tuple[int, np.ndarray]:
        """Select action using MCTS search from current observation.

        Args:
            observation: Current game state observation.
            training: If True, sample from action probabilities; if False, take best action.

        Returns:
            A tuple containing the selected action index and the MCTS policy probabilities.
        """
        log.debug(f"Selecting action. Training mode: {training}")

        obs_tensor = jax.device_put(jnp.array(observation)[None], self.device)  # [None] adds batch dimension

        hidden_state, _, _ = self.network.initial_inference(obs_tensor)
        hidden_state_array = jnp.asarray(hidden_state, dtype=jnp.float32)[0]  # Remove batch dimension

        # Use MCTS to get action probabilities
        action_probs = self.mcts.search(root_hidden_state=hidden_state_array, add_noise=training)

        if training:
            # Sample action from probabilities
            action_idx = int(jax.random.choice(key.get(), len(action_probs), p=action_probs))
        else:
            # Take best action
            action_idx = int(np.argmax(action_probs))

        log.debug(f"Selected action: {action_idx}")
        return int(action_idx), action_probs

    def update_target_network(self) -> None:
        """Update the target network's weights with the main network's weights."""
        log.debug("Updating target network.")
        online_params = nnx.state(self.network, nnx.Param)
        nnx.update(self.target_network, online_params)

    def save_state(self) -> Dict[str, Any]:
        """Get the current state of the agent for checkpointing.

        Returns:
            A dictionary containing the network and optimizer state.
        """
        log.debug("Saving agent state.")
        return {
            "network_state": nnx.state(self.network),
            "optimizer_state": self.optimizer_state,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Load the agent's state from a dictionary.

        Args:
            state: A dictionary containing the network and optimizer state.
        """
        log.info("Loading agent state from dictionary.")
        nnx.update(self.network, state["network_state"])
        self.optimizer_state = state["optimizer_state"]

        log.info("Updating target network after loading state.")
        self.update_target_network()
        log.info("Agent state loaded successfully.")
