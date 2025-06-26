"""Configuration for CuMind."""

import dataclasses
import json
from pathlib import Path
from typing import Tuple

import chex


@chex.dataclass
class Config:
    """Hyperparameter configuration for the CuMind agent.

    This dataclass defines all configurable hyperparameters. For more details
    on tuning, see the project's documentation.
    """

    # Network architecture
    hidden_dim: int = 64
    num_blocks: int = 4

    # Training
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    target_update_frequency: int = 200
    checkpoint_interval: int = 50

    # MCTS
    num_simulations: int = 50
    c_puct: float = 1.0
    dirichlet_alpha: float = 0.3
    exploration_fraction: float = 0.25

    # Environment
    action_space_size: int = 4
    observation_shape: Tuple[int, ...] = (84, 84, 3)

    # Self-Play
    num_unroll_steps: int = 5
    td_steps: int = 10
    discount: float = 0.997

    # Replay Buffer
    replay_buffer_size: int = 10000
    min_replay_size: int = 1000
    min_replay_fill_pct: float = 0.1

    @classmethod
    def from_json(cls, json_path: str) -> "Config":
        """Loads a configuration from a JSON file.

        Args:
            json_path: Path to the JSON configuration file.

        Returns:
            A Config instance with the loaded parameters.
        """
        json_file = Path(json_path)
        if not json_file.exists():
            raise FileNotFoundError(f"Config file not found: {json_path}")

        with open(json_file, "r", encoding="utf-8") as f:
            config_dict = json.load(f)

        return cls(**config_dict)

    def to_json(self, json_path: str) -> None:
        """Saves the configuration to a JSON file.

        Args:
            json_path: Path to save the JSON configuration file.
        """
        json_file = Path(json_path)
        json_file.parent.mkdir(parents=True, exist_ok=True)

        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(dataclasses.asdict(self), f, indent=2)

    def validate(self) -> None:
        """Validates the configuration parameters.

        Raises:
            ValueError: If any parameter is found to be invalid.
        """
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}")
        if self.action_space_size <= 0:
            raise ValueError(f"action_space_size must be positive, got {self.action_space_size}")
        if not self.observation_shape:
            raise ValueError("observation_shape cannot be empty")
        if len(self.observation_shape) not in [1, 3]:
            raise ValueError(f"observation_shape must be 1D or 3D, got {len(self.observation_shape)}D")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.num_simulations <= 0:
            raise ValueError(f"num_simulations must be positive, got {self.num_simulations}")
