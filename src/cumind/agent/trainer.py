"""Training loop implementation."""

from typing import Any, Dict, List

import chex
import jax
import jax.numpy as jnp
import optax  # type: ignore

from ..config import Config
from ..core.network import CuMindNetwork
from .agent import Agent


class Trainer:
    """Training loop: sampling, unrolling, loss computation."""

    def __init__(self, config: Config):
        """Initialize trainer with config and network components.

        Args:
            config: CuMind configuration with training parameters

        Implementation:
            - Create CuMindNetwork instance
            - Setup optimizer with config learning rate and weight decay
            - Initialize training state and counters
        """
        # Branch: feature/trainer-init
        raise NotImplementedError("Trainer.__init__ needs to be implemented")

    def _get_network_parameters(self) -> Any:
        """Get network parameters for optimization.

        Returns:
            Network parameters iterator

        Implementation:
            - Return network.parameters() for optimizer
        """
        # Branch: feature/get-network-parameters
        raise NotImplementedError("Trainer._get_network_parameters needs to be implemented")

    def train_step(self, batch: List[Any]) -> Dict[str, float]:
        """Perform one training step on a batch of data.

        Args:
            batch: List of training trajectories

        Returns:
            Dictionary of loss values for monitoring

        Implementation:
            - Extract observations, actions, targets from batch
            - Run forward pass through network
            - Compute CuMind losses (value, policy, reward)
            - Perform backward pass and optimizer step
            - Return loss metrics as dict
        """
        # Branch: feature/trainer-step
        raise NotImplementedError("Trainer.train_step needs to be implemented")

    def save_checkpoint(self, path: str) -> None:
        """Save training checkpoint to disk.

        Args:
            path: File path to save checkpoint

        Implementation:
            - Create checkpoint dict with network state, optimizer state
            - Use flax.training.checkpoints.save_checkpoint with msgpack serialization
            - Include training step counter and config
        """
        # Branch: feature/trainer-save
        raise NotImplementedError("Trainer.save_checkpoint needs to be implemented")

    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint from disk.

        Args:
            path: File path to load checkpoint from

        Implementation:
            - Use flax.training.checkpoints.restore_checkpoint to read checkpoint
            - Restore network and optimizer state using from_state_dict
            - Restore training step counter
        """
        # Branch: feature/trainer-load
        raise NotImplementedError("Trainer.load_checkpoint needs to be implemented")
