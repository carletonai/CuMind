"""CuMind agent implementation."""

from typing import Any, Dict, List, Tuple, cast

import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax  # type: ignore
from flax import nnx

from ..config import Config
from ..core.mcts import MCTS
from ..core.network import CuMindNetwork
from ..utils.checkpoint import load_checkpoint, save_checkpoint


class Agent:
    """CuMind agent for training and inference."""

    def __init__(self, config: Config, existing_state: Dict[str, Any] | None = None):
        """Initialize CuMind agent with network, optimizer, and MCTS.

        Args:
            config: Config with network architecture and training parameters.
            existing_state: Optional dictionary to load agent state from.
        """
        self.config = config

        # Create network with random initialization
        key = jax.random.PRNGKey(0)
        rngs = nnx.Rngs(params=key)

        self.network = CuMindNetwork(observation_shape=config.observation_shape, action_space_size=config.action_space_size, hidden_dim=config.hidden_dim, num_blocks=config.num_blocks, rngs=rngs)

        # Setup optimizer
        self.optimizer = optax.adamw(learning_rate=config.learning_rate, weight_decay=config.weight_decay)

        if existing_state:
            self.load_state(existing_state)
        else:
            self.optimizer_state = self.optimizer.init(nnx.state(self.network, nnx.Param))

        # Initialize MCTS
        self.mcts = MCTS(self.network, config)

    def select_action(self, observation: np.ndarray, training: bool = False) -> Tuple[int, np.ndarray]:
        """Select action using MCTS search from current observation.

        Args:
            observation: Current game state observation.
            training: If True, sample from action probabilities; if False, take best action.

        Returns:
            A tuple containing the selected action index and the MCTS policy probabilities.
        """
        # Convert observation to tensor and get initial hidden state
        obs_tensor = jnp.array(observation)[None]  # Add batch dimension
        hidden_state, _, _ = self.network.initial_inference(obs_tensor)
        hidden_state_array = jnp.asarray(hidden_state, dtype=jnp.float32)[0]  # Remove batch dimension

        # Use MCTS to get action probabilities
        action_probs = self.mcts.search(root_hidden_state=hidden_state_array, add_noise=training)

        if training:
            # Sample action from probabilities
            action_idx = np.random.choice(len(action_probs), p=action_probs)
        else:
            # Take best action
            action_idx = int(np.argmax(action_probs))

        return int(action_idx), action_probs

    def save_state(self) -> Dict[str, Any]:
        """Get the current state of the agent for checkpointing.

        Returns:
            A dictionary containing the network and optimizer state.
        """
        return {
            "network_state": nnx.state(self.network),
            "optimizer_state": self.optimizer_state,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Load the agent's state from a dictionary.

        Args:
            state: A dictionary containing the network and optimizer state.
        """
        nnx.update(self.network, state["network_state"])
        self.optimizer_state = state["optimizer_state"]
