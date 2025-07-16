"""Configuration for CuMind."""

import dataclasses
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import chex

from .utils.logger import log


@chex.dataclass
class Config:
    """Hyperparameter configuration for the CuMind agent.

    This dataclass defines all configurable hyperparameters. For more details
    on tuning, see the project's documentation.
    """

    # Network architecture
    network_hidden_dim: int = 128
    network_num_blocks: int = 2
    network_conv_channels: int = 32

    # Training
    training_batch_size: int = 64
    training_learning_rate: float = 0.01
    training_weight_decay: float = 0.0001
    training_target_update_frequency: int = 250
    training_checkpoint_interval: int = 50
    training_num_episodes: int = 1220
    training_train_frequency: int = 2
    training_checkpoint_root_dir: str = "checkpoints"

    # MCTS
    mcts_num_simulations: int = 25
    mcts_c_puct: float = 1.25
    mcts_dirichlet_alpha: float = 0.25
    mcts_exploration_fraction: float = 0.25

    # Environment
    env_name: str = "CartPole-v1"
    env_action_space_size: int = 2
    env_observation_shape: Tuple[int, ...] = (4,)

    # Self-Play
    selfplay_num_unroll_steps: int = 5
    selfplay_td_steps: int = 10
    selfplay_discount: float = 0.997

    # Memory
    memory_capacity: int = 2000
    memory_min_size: int = 100
    memory_min_pct: float = 0.1
    memory_per_alpha: float = 0.6
    # Tree

    memory_per_epsilon: float = 1e-6
    # Prioritized Buffer
    memory_per_beta: float = 0.4

    # Data Types
    dtypes_model: str = "float32"
    dtypes_action: str = "int32"
    dtypes_target: str = "float32"

    # Devices
    device: str = "cpu"
    seed: int = 42

    @classmethod
    def from_json(cls, json_path: str) -> "Config":
        """Loads a configuration from a JSON file.
        Args:
            json_path: Path to the JSON configuration file.
        Returns:
            A Config instance with the loaded parameters.
        """
        log.info(f"Loading configuration from {json_path}...")
        json_file = Path(json_path)
        if not json_file.exists():
            log.critical(f"Config file not found: {json_path}")
            raise FileNotFoundError(f"Config file not found: {json_path}")

        with open(json_file, "r", encoding="utf-8") as f:
            config_dict = json.load(f)

        if "CuMind" not in config_dict:
            log.critical("Root key 'CuMind' not found in config file.")
            raise ValueError("Root key 'CuMind' not found in config file.")

        config_params = config_dict["CuMind"]
        flattened_dict = {}
        for section, params in config_params.items():
            if isinstance(params, dict):
                for key, value in params.items():
                    flattened_dict[f"{section}_{key}"] = value
            else:
                flattened_dict[section] = params

        log.info("Configuration loaded successfully.")
        return cls(**flattened_dict)

    def _format_json(self, obj: Any, indent: int = 0) -> str:
        """Format JSON with open brace style and one-liner arrays."""
        if isinstance(obj, dict):
            if not obj:
                return "{}"

            items = []
            for key, value in obj.items():
                key_str = json.dumps(key)
                if isinstance(value, (dict, list)) and value:
                    value_str = self._format_json(value, indent + 2)
                    items.append(f'{" " * (indent + 2)}{key_str}:\n{" " * (indent + 2)}{value_str}')
                else:
                    value_str = self._format_json(value, indent + 2)
                    items.append(f'{" " * (indent + 2)}{key_str}: {value_str}')

            return "{\n" + ",\n".join(items) + "\n" + " " * indent + "}"

        elif isinstance(obj, list):
            if not obj:
                return "[]"

            if all(isinstance(item, (str, int, float, bool)) or item is None for item in obj):
                return json.dumps(obj)

            items = []
            for item in obj:
                items.append(self._format_json(item, indent + 2))

            return "[\n" + ",\n".join(f'{" " * (indent + 2)}{item}' for item in items) + "\n" + " " * indent + "]"

        else:
            return json.dumps(obj)

    def to_json(self, json_path: str) -> None:
        """Saves the configuration to a JSON file.
        Args:
            json_path: Path to save the JSON configuration file.
        """
        log.info(f"Saving configuration to {json_path}...")
        json_file = Path(json_path)
        json_file.parent.mkdir(parents=True, exist_ok=True)

        config_dict = dataclasses.asdict(self)
        organized_config: Dict[str, Any] = {}

        for key, value in config_dict.items():
            if "_" in key:
                section, field = key.split("_", 1)
                if section not in organized_config:
                    organized_config[section] = {}
                organized_config[section][field] = value
            else:
                organized_config[key] = value

        final_config = {"CuMind": organized_config}

        with open(json_file, "w", encoding="utf-8") as f:
            f.write(self._format_json(final_config))
            f.write("\n")

        log.info("Configuration saved successfully.")

    def validate(self) -> None:
        """Validates the configuration parameters.
        Raises:
            ValueError: If any parameter is found to be invalid.
        """
        log.info("Validating configuration...")
        if self.network_hidden_dim <= 0:
            log.critical(f"Invalid hidden_dim: {self.network_hidden_dim}. Must be positive.")
            raise ValueError(f"hidden_dim must be positive, got {self.network_hidden_dim}")
        if self.env_action_space_size <= 0:
            log.critical(f"Invalid action_space_size: {self.env_action_space_size}. Must be positive.")
            raise ValueError(f"action_space_size must be positive, got {self.env_action_space_size}")
        if not self.env_observation_shape:
            log.critical("observation_shape cannot be empty.")
            raise ValueError("observation_shape cannot be empty")
        if len(self.env_observation_shape) not in [1, 3]:
            log.critical(f"Unsupported observation_shape dimensionality: {len(self.env_observation_shape)}D. Must be 1D or 3D.")
            raise ValueError(f"observation_shape must be 1D or 3D, got {len(self.env_observation_shape)}D")
        if self.training_learning_rate <= 0:
            log.critical(f"Invalid learning_rate: {self.training_learning_rate}. Must be positive.")
            raise ValueError(f"learning_rate must be positive, got {self.training_learning_rate}")
        if self.training_batch_size <= 0:
            log.critical(f"Invalid batch_size: {self.training_batch_size}. Must be positive.")
            raise ValueError(f"batch_size must be positive, got {self.training_batch_size}")
        if self.mcts_num_simulations <= 0:
            log.critical(f"Invalid num_simulations: {self.mcts_num_simulations}. Must be positive.")
            raise ValueError(f"num_simulations must be positive, got {self.mcts_num_simulations}")

        valid_model_dtypes = ["float32", "float16", "bfloat16"]
        if self.dtypes_model not in valid_model_dtypes:
            log.critical(f"Invalid model_dtype: {self.dtypes_model}. Must be one of {valid_model_dtypes}.")
            raise ValueError(f"model_dtype must be one of {valid_model_dtypes}, got {self.dtypes_model}")

        valid_action_dtypes = ["int32", "int64"]
        if self.dtypes_action not in valid_action_dtypes:
            log.critical(f"Invalid action_dtype: {self.dtypes_action}. Must be one of {valid_action_dtypes}.")
            raise ValueError(f"action_dtype must be one of {valid_action_dtypes}, got {self.dtypes_action}")

        valid_target_dtypes = ["float32", "float16", "bfloat16"]
        if self.dtypes_target not in valid_target_dtypes:
            log.critical(f"Invalid target_dtype: {self.dtypes_target}. Must be one of {valid_target_dtypes}.")
            raise ValueError(f"target_dtype must be one of {valid_target_dtypes}, got {self.dtypes_target}")

        valid_device = ["cpu", "gpu", "tpu"]
        if self.device not in valid_device:
            log.critical(f"Invalid device_type: {self.device}. Must be one of {valid_device}.")
            raise ValueError(f"device_type must be one of {valid_device}, got {self.device}")

        log.info("Configuration validation successful.")
