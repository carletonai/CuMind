"""Global configuration management for CuMind."""

import dataclasses
import inspect
import json
import re
from pathlib import Path
from typing import Any, Tuple

from .logger import log
from .resolve import resolve


@dataclasses.dataclass(frozen=True)
class RepresentationConfig:
    """Configuration for representation network."""

    type: str = "cumind.core.resnet.ResNet"
    hidden_dim: int = 128
    num_blocks: int = 2
    conv_channels: int = 32

    def __call__(self, **kwargs):
        """Instantiate the representation network."""
        cls = resolve(self.type)
        params = {k: v for k, v in dataclasses.asdict(self).items() if k != "type"}
        sig = inspect.signature(cls)
        valid = {k: v for k, v in {**params, **kwargs}.items() if k in sig.parameters}
        return cls(**valid)


@dataclasses.dataclass(frozen=True)
class DynamicsConfig:
    """Configuration for dynamics network."""

    type: str = "cumind.core.mlp.MLPWithEmbedding"
    hidden_dim: int = 128
    num_blocks: int = 2

    def __call__(self, **kwargs):
        """Instantiate the representation network."""
        cls = resolve(self.type)
        params = {k: v for k, v in dataclasses.asdict(self).items() if k != "type"}
        sig = inspect.signature(cls)
        valid = {k: v for k, v in {**params, **kwargs}.items() if k in sig.parameters}
        return cls(**valid)


@dataclasses.dataclass(frozen=True)
class PredictionConfig:
    """Configuration for prediction network."""

    type: str = "cumind.core.mlp.MLPDual"
    hidden_dim: int = 128
    num_blocks: int = 2

    def __call__(self, **kwargs):
        """Instantiate the representation network."""
        cls = resolve(self.type)
        params = {k: v for k, v in dataclasses.asdict(self).items() if k != "type"}
        sig = inspect.signature(cls)
        valid = {k: v for k, v in {**params, **kwargs}.items() if k in sig.parameters}
        return cls(**valid)


@dataclasses.dataclass(frozen=True)
class MemoryConfig:
    """Configuration for memory buffer."""

    type: str = "cumind.data.memory.MemoryBuffer"
    capacity: int = 2000
    min_size: int = 100
    min_pct: float = 0.1
    per_alpha: float = 0.6
    per_epsilon: float = 1e-6
    per_beta: float = 0.4

    def __call__(self, **kwargs):
        """Instantiate the representation network."""
        cls = resolve(self.type)
        params = {k: v for k, v in dataclasses.asdict(self).items() if k != "type"}
        sig = inspect.signature(cls)
        valid = {k: v for k, v in {**params, **kwargs}.items() if k in sig.parameters}
        return cls(**valid)


@dataclasses.dataclass(frozen=True)
class TrainingConfig:
    """Configuration for training."""

    optimizer: str = "optax.adamw"
    batch_size: int = 64
    learning_rate: float = 0.01
    weight_decay: float = 0.0001
    target_update_frequency: int = 250
    checkpoint_interval: int = 50
    num_episodes: int = 1220
    train_frequency: int = 2
    checkpoint_root_dir: str = "checkpoints"


@dataclasses.dataclass(frozen=True)
class MCTSConfig:
    """Configuration for MCTS."""

    num_simulations: int = 25
    c_puct: float = 1.25
    dirichlet_alpha: float = 0.25
    exploration_fraction: float = 0.25


@dataclasses.dataclass(frozen=True)
class EnvironmentConfig:
    """Configuration for environment."""

    name: str = "CartPole-v1"
    action_space_size: int = 2
    observation_shape: Tuple[int, ...] = (4,)


@dataclasses.dataclass(frozen=True)
class SelfPlayConfig:
    """Configuration for self-play."""

    num_unroll_steps: int = 5
    td_steps: int = 10
    discount: float = 0.997


@dataclasses.dataclass(frozen=True)
class DataTypesConfig:
    """Configuration for data types."""

    model: str = "float32"
    action: str = "int32"
    target: str = "float32"


@dataclasses.dataclass(frozen=True)
class Configuration:
    """Main configuration for CuMind."""

    # Hot-swappable modules
    representation: RepresentationConfig = dataclasses.field(default_factory=RepresentationConfig)
    dynamics: DynamicsConfig = dataclasses.field(default_factory=DynamicsConfig)
    prediction: PredictionConfig = dataclasses.field(default_factory=PredictionConfig)
    memory: MemoryConfig = dataclasses.field(default_factory=MemoryConfig)

    # Other sections
    training: TrainingConfig = dataclasses.field(default_factory=TrainingConfig)
    mcts: MCTSConfig = dataclasses.field(default_factory=MCTSConfig)
    env: EnvironmentConfig = dataclasses.field(default_factory=EnvironmentConfig)
    selfplay: SelfPlayConfig = dataclasses.field(default_factory=SelfPlayConfig)
    dtypes: DataTypesConfig = dataclasses.field(default_factory=DataTypesConfig)

    # Global settings
    device: str = "cpu"
    seed: int = 42

    @classmethod
    def _from_json(cls, json_path: str) -> "Configuration":
        """Loads a configuration from a JSON file."""
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

        # Create section configs
        section_configs = {}
        for section, params in config_params.items():
            if section == "representation":
                section_configs[section] = RepresentationConfig(**params)
            elif section == "dynamics":
                section_configs[section] = DynamicsConfig(**params)
            elif section == "prediction":
                section_configs[section] = PredictionConfig(**params)
            elif section == "memory":
                section_configs[section] = MemoryConfig(**params)
            elif section == "training":
                section_configs[section] = TrainingConfig(**params)
            elif section == "mcts":
                section_configs[section] = MCTSConfig(**params)
            elif section == "env":
                section_configs[section] = EnvironmentConfig(**params)
            elif section == "selfplay":
                section_configs[section] = SelfPlayConfig(**params)
            elif section == "dtypes":
                section_configs[section] = DataTypesConfig(**params)
            else:
                section_configs[section] = params

        log.info("Configuration loaded successfully.")
        return cls(**section_configs)

    def _to_json(self, json_path: str) -> None:
        """Saves the configuration to a JSON file."""
        log.info(f"Saving configuration to {json_path}...")
        json_file = Path(json_path)
        json_file.parent.mkdir(parents=True, exist_ok=True)

        config_dict = dataclasses.asdict(self)
        organized_config = {}

        for key, value in config_dict.items():
            if hasattr(value, "__dataclass_fields__"):
                # It's a dataclass, convert to dict
                organized_config[key] = dataclasses.asdict(value)
            else:
                organized_config[key] = value

        final_config = {"CuMind": organized_config}

        # Dump to string first
        json_str = json.dumps(final_config, indent=2)
        # Compact single-element lists: [\n  4\n] -> [4]
        json_str = re.sub(r"\[\s*([\d.eE+-]+)\s*\]", r"[\1]", json_str)

        with open(json_file, "w", encoding="utf-8") as f:
            f.write(json_str)
            f.write("\n")

        log.info("Configuration saved successfully.")

    def validate(self) -> None:
        """Validates the configuration parameters."""
        log.info("Validating configuration...")

        # Validate representation
        if self.representation.hidden_dim <= 0:
            raise ValueError(f"representation.hidden_dim must be positive, got {self.representation.hidden_dim}")

        # Validate environment
        if self.env.action_space_size <= 0:
            raise ValueError(f"env.action_space_size must be positive, got {self.env.action_space_size}")
        if not self.env.observation_shape:
            raise ValueError("env.observation_shape cannot be empty")
        if len(self.env.observation_shape) not in [1, 3]:
            raise ValueError(f"env.observation_shape must be 1D or 3D, got {len(self.env.observation_shape)}D")

        # Validate training
        if self.training.learning_rate <= 0:
            raise ValueError(f"training.learning_rate must be positive, got {self.training.learning_rate}")
        if self.training.batch_size <= 0:
            raise ValueError(f"training.batch_size must be positive, got {self.training.batch_size}")

        # Validate MCTS
        if self.mcts.num_simulations <= 0:
            raise ValueError(f"mcts.num_simulations must be positive, got {self.mcts.num_simulations}")

        # Validate data types
        valid_model_dtypes = ["float32", "float16", "bfloat16"]
        if self.dtypes.model not in valid_model_dtypes:
            raise ValueError(f"dtypes.model must be one of {valid_model_dtypes}, got {self.dtypes.model}")

        valid_action_dtypes = ["int32", "int64"]
        if self.dtypes.action not in valid_action_dtypes:
            raise ValueError(f"dtypes.action must be one of {valid_action_dtypes}, got {self.dtypes.action}")

        valid_target_dtypes = ["float32", "float16", "bfloat16"]
        if self.dtypes.target not in valid_target_dtypes:
            raise ValueError(f"dtypes.target must be one of {valid_target_dtypes}, got {self.dtypes.target}")

        # Validate device
        valid_devices = ["cpu", "gpu", "tpu"]
        if self.device not in valid_devices:
            raise ValueError(f"device must be one of {valid_devices}, got {self.device}")

        log.info("Configuration validation successful.")

    def __getattr__(self, field: str) -> Any:
        """
        Custom attribute access for config fields. Raises a clear exception if the field does not exist.
        This allows for runtime type checking and custom error messages for missing config fields.
        """
        if field not in self.__dataclass_fields__:
            raise AttributeError(f"Config has no field '{field}' (available fields: {list(self.__dataclass_fields__.keys())})")
        return self.__getattribute__(field)

    @classmethod
    def load(cls, path: str) -> None:
        """Load configuration from a JSON file and set as singleton instance."""
        global cfg
        cfg = cls._from_json(path)

    @classmethod
    def save(cls, path: str) -> None:
        """Save configuration to a JSON file."""
        global cfg
        if cfg is None:
            raise RuntimeError("Configuration not loaded. Call Config.load(path) or Config.set(config) first.")
        cfg._to_json(path)

    @classmethod
    def set(cls, config: "Configuration") -> None:
        """Set the singleton config instance directly."""
        global cfg
        cfg = config

    @classmethod
    def get(cls) -> "Configuration":
        """Get the current singleton config instance, or raise if not set."""
        global cfg
        if cfg is None:
            raise RuntimeError("Configuration not loaded. Call Config.load(path) or Config.set(config) first.")
        return cfg


# Singleton instance for configuration
cfg: Configuration = Configuration()
