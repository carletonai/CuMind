"""Singleton configuration manager for CuMind."""

import dataclasses
import json
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from ..config import Config
from .export import str_to_nnx, validate_networks
from .logger import log
from .prng import key


class ConfigManager:
    """Singleton configuration manager for CuMind."""

    _instance: Optional["ConfigManager"] = None
    _lock = threading.RLock()
    _initialized: bool = False

    def __new__(cls) -> "ConfigManager":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._config: Optional[Config] = None
        self._initialized = True
        log.debug("ConfigManager singleton instance created")
        key.seed(0)  # gentle reminder to reseed after config validation

    @classmethod
    def instance(cls) -> "ConfigManager":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance."""
        if cls._instance is not None:
            cls._instance.reset_config()

    def reset_config(self) -> None:
        """Reset the current configuration to its default state."""
        self._config = None

    def from_json(self, json_path: str) -> None:
        """Load configuration from a JSON file.

        Args:
            json_path: Path to the JSON configuration file.
        """
        if self._config is not None:
            log.warning("Configuration already loaded. Ignoring new configuration.")
            return

        try:
            log.info(f"Loading configuration from {json_path}...")
            json_file = Path(json_path)
            if not json_file.exists():
                log.critical(f"Config file not found: {json_path}")
                raise FileNotFoundError(f"Config file not found: {json_path}")

            with open(json_file, "r", encoding="utf-8") as f:
                config_dict = json.load(f)

            flattened_dict = {}
            for section, params in config_dict.items():
                if isinstance(params, dict):
                    flattened_dict.update(params)
                else:
                    flattened_dict[section] = params

            self._config = Config(**flattened_dict)
            log.info("Configuration loaded successfully.")

            self.validate()
        except Exception as e:
            log.critical(f"Failed to load configuration from {json_path}: {e}")
            self._config = None
            raise

    def to_json(self, json_path: str) -> None:
        """Save the configuration to a JSON file.

        Args:
            json_path: Path to save the JSON configuration file.
        """
        cfg = self.get()
        log.info(f"Saving configuration to {json_path}...")
        json_file = Path(json_path)
        json_file.parent.mkdir(parents=True, exist_ok=True)

        section_mapping = {
            "Agent": ["representation_network", "dynamics_network", "prediction_network"],
            "Network architecture": ["network_architecture"],
            "Training": ["batch_size", "learning_rate", "weight_decay", "target_update_frequency", "checkpoint_interval", "num_episodes", "train_frequency", "checkpoint_root_dir"],
            "MCTS": ["num_simulations", "c_puct", "dirichlet_alpha", "exploration_fraction"],
            "Environment": ["env_name", "action_space_size", "observation_shape"],
            "Self-Play": ["num_unroll_steps", "td_steps", "discount"],
            "Memory": ["memory_capacity", "min_memory_size", "min_memory_pct", "per_alpha", "per_epsilon", "per_beta"],
            "Data Types": ["model_dtype", "action_dtype", "target_dtype"],
            "Device": ["device_type"],
            "Other": ["seed"],
        }

        config_dict = dataclasses.asdict(cfg)
        organized_config = {}

        for section, fields in section_mapping.items():
            section_data = {}
            for field in fields:
                if field in config_dict:
                    section_data[field] = config_dict[field]
            if section_data:
                organized_config[section] = section_data

        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(organized_config, f, indent=2)
        log.info("Configuration saved successfully.")

    def validate(self) -> None:
        """Validate the configuration parameters."""
        cfg = self.get()
        log.info("Validating configuration...")

        if cfg.action_space_size <= 0:
            log.critical(f"Invalid action_space_size: {cfg.action_space_size}. Must be positive.")
            raise ValueError(f"action_space_size must be positive, got {cfg.action_space_size}")
        if not cfg.observation_shape:
            log.critical("observation_shape cannot be empty.")
            raise ValueError("observation_shape cannot be empty")
        if len(cfg.observation_shape) not in [1, 3]:
            log.critical(f"Unsupported observation_shape dimensionality: {len(cfg.observation_shape)}D. Must be 1D or 3D.")
            raise ValueError(f"observation_shape must be 1D or 3D, got {len(cfg.observation_shape)}D")
        if cfg.learning_rate <= 0:
            log.critical(f"Invalid learning_rate: {cfg.learning_rate}. Must be positive.")
            raise ValueError(f"learning_rate must be positive, got {cfg.learning_rate}")
        if cfg.batch_size <= 0:
            log.critical(f"Invalid batch_size: {cfg.batch_size}. Must be positive.")
            raise ValueError(f"batch_size must be positive, got {cfg.batch_size}")
        if cfg.num_simulations <= 0:
            log.critical(f"Invalid num_simulations: {cfg.num_simulations}. Must be positive.")
            raise ValueError(f"num_simulations must be positive, got {cfg.num_simulations}")

        valid_model_dtypes = ["float32", "float16", "bfloat16"]
        if cfg.model_dtype not in valid_model_dtypes:
            log.critical(f"Invalid model_dtype: {cfg.model_dtype}. Must be one of {valid_model_dtypes}.")
            raise ValueError(f"model_dtype must be one of {valid_model_dtypes}, got {cfg.model_dtype}")

        valid_action_dtypes = ["int32", "int64"]
        if cfg.action_dtype not in valid_action_dtypes:
            log.critical(f"Invalid action_dtype: {cfg.action_dtype}. Must be one of {valid_action_dtypes}.")
            raise ValueError(f"action_dtype must be one of {valid_action_dtypes}, got {cfg.action_dtype}")

        valid_target_dtypes = ["float32", "float16", "bfloat16"]
        if cfg.target_dtype not in valid_target_dtypes:
            log.critical(f"Invalid target_dtype: {cfg.target_dtype}. Must be one of {valid_target_dtypes}.")
            raise ValueError(f"target_dtype must be one of {valid_target_dtypes}, got {cfg.target_dtype}")

        valid_device_types = ["cpu", "gpu", "tpu"]
        if cfg.device_type not in valid_device_types:
            log.critical(f"Invalid device_type: {cfg.device_type}. Must be one of {valid_device_types}.")
            raise ValueError(f"device_type must be one of {valid_device_types}, got {cfg.device_type}")

        validate_networks(cfg)
        log.info("Configuration validation successful.")

    def load(self, config_obj: Optional[Config] = None, json_path: Optional[str] = None) -> None:
        """Load configuration from Config object or JSON file."""
        if self._config is not None:
            log.warning("Configuration already loaded. Ignoring new configuration.")
            return

        try:
            if config_obj is not None:
                self._config = config_obj
                log.info("Configuration loaded from Config object")
            elif json_path is not None:
                self.from_json(json_path)
                return
            else:
                self._config = Config()
                log.info("Configuration loaded with default values")

            self.validate()
        except Exception as e:
            log.critical(f"Failed to load configuration: {e}")
            self._config = None
            raise

    def get(self) -> Config:
        """Get current configuration."""
        if self._config is None:
            log.warning("Configuration not loaded. Loading default configuration.")
            self._config = Config()
            self.validate()
        return self._config

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the config object."""
        if name.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        cfg = self.get()
        if hasattr(cfg, name):
            return getattr(cfg, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


# Singleton instance
config = ConfigManager.instance()
