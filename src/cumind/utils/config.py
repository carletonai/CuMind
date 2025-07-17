"""Global configuration management for CuMind."""

import dataclasses
import inspect
import json
import re
import threading
from pathlib import Path
from typing import Any, Optional, Tuple, Type, Union, get_args, get_origin
from typing import Dict as DictType

from flax import nnx

from .logger import log
from .prng import key
from .resolve import resolve


@dataclasses.dataclass(frozen=True)
class HotSwappableConfig:
    """Base class for hot-swappable module configurations."""

    type: Optional[Union[str, Type[Any]]] = None

    def __call__(self, *args, **kwargs):
        if self.type is None:
            raise ValueError("Type must be specified for hot-swappable config")

        cls = resolve(self.type) if isinstance(self.type, str) else self.type

        field_params = {k: v for k, v in dataclasses.asdict(self).items() if k != "type"}
        extra_params = self.extras()
        all_params = {**field_params, **extra_params}

        # Get constructor signature to know which parameters to pass
        sig = inspect.signature(cls.__init__)
        constructor_params = list(sig.parameters.keys())
        filtered_params = {k: v for k, v in all_params.items() if k in constructor_params and k != "self"}

        try:
            return cls(*args, **filtered_params, **kwargs)
        except Exception as e:
            log.exception(f"Failed to instantiate {cls.__name__} with params: {filtered_params}. {e}")
            raise

    def extras(self) -> DictType[str, Any]:
        """Override this in subclasses to add global cfg values."""
        return {}


@dataclasses.dataclass(frozen=True)
class RepresentationConfig(HotSwappableConfig):
    """Configuration for representation network."""

    type: Optional[Union[str, Type[Any]]] = "cumind.core.resnet.ResNet"
    num_blocks: int = 2
    conv_channels: int = 32
    hidden_dim: int = 128

    def extras(self) -> DictType[str, Any]:
        input_shape = cfg.env.observation_shape
        rngs = nnx.Rngs(params=key())
        return locals()


@dataclasses.dataclass(frozen=True)
class DynamicsConfig(HotSwappableConfig):
    """Configuration for dynamics network."""

    type: Optional[Union[str, Type[Any]]] = "cumind.core.mlp.MLPWithEmbedding"
    num_blocks: int = 2

    def extras(self) -> DictType[str, Any]:
        hidden_dim = cfg.networks.hidden_dim
        embedding_size = cfg.env.action_space_size
        rngs = nnx.Rngs(params=key())
        return locals()


@dataclasses.dataclass(frozen=True)
class PredictionConfig(HotSwappableConfig):
    """Configuration for prediction network."""

    type: Optional[Union[str, Type[Any]]] = "cumind.core.mlp.MLPDual"

    def extras(self) -> DictType[str, Any]:
        hidden_dim = cfg.networks.hidden_dim
        output_size = cfg.env.action_space_size
        rngs = nnx.Rngs(params=key())
        return locals()


@dataclasses.dataclass(frozen=True)
class MemoryConfig(HotSwappableConfig):
    """Configuration for memory buffer."""

    type: Optional[Union[str, Type[Any]]] = "cumind.data.memory.MemoryBuffer"
    capacity: int = 2000
    min_size: int = 100
    min_pct: float = 0.1
    per_alpha: float = 0.6
    per_epsilon: float = 1e-6
    per_beta: float = 0.4

    def extras(self) -> DictType[str, Any]:
        return {}


@dataclasses.dataclass(frozen=True)
class GeneralNetworksConfig:
    """Configuration for general networks."""

    hidden_dim: int = 128


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
    _set_lock = threading.RLock()
    """Main configuration for CuMind."""

    # Hot-swappable modules
    representation: RepresentationConfig = dataclasses.field(default_factory=RepresentationConfig)
    dynamics: DynamicsConfig = dataclasses.field(default_factory=DynamicsConfig)
    prediction: PredictionConfig = dataclasses.field(default_factory=PredictionConfig)
    memory: MemoryConfig = dataclasses.field(default_factory=MemoryConfig)

    # Other sections
    networks: GeneralNetworksConfig = dataclasses.field(default_factory=GeneralNetworksConfig)
    training: TrainingConfig = dataclasses.field(default_factory=TrainingConfig)
    mcts: MCTSConfig = dataclasses.field(default_factory=MCTSConfig)
    env: EnvironmentConfig = dataclasses.field(default_factory=EnvironmentConfig)
    selfplay: SelfPlayConfig = dataclasses.field(default_factory=SelfPlayConfig)
    dtypes: DataTypesConfig = dataclasses.field(default_factory=DataTypesConfig)

    # Global settings
    device: str = "cpu"
    seed: int = 42

    def __post_init__(self):
        key.seed(self.seed)
        self._validate()

    def __getattr__(self, field: str) -> Any:
        """Custom attribute access for config fields with clear error if missing."""
        if field not in self.__dataclass_fields__:
            log.critical(f"Config has no field '{field}' (available fields: {list(self.__dataclass_fields__.keys())})")
            raise AttributeError(f"Config has no field '{field}' (available fields: {list(self.__dataclass_fields__.keys())})")
        return self.__getattribute__(field)

    @classmethod
    def load(cls, path: str) -> None:
        """Load configuration from a JSON file and set as singleton instance. Automatically validates after loading."""
        global cfg
        log.info(f"Loading configuration from {path}")
        cfg = cls._from_json(path)
        try:
            cfg._validate()
            log.info("Configuration validated successfully after loading.")
        except Exception as e:
            log.exception(f"Configuration validation failed after loading: {e}")
            raise

    @classmethod
    def save(cls, path: str) -> None:
        """Save configuration to a JSON file."""
        global cfg
        log.info(f"Saving configuration to {path}")
        cfg._to_json(path)
        log.info("Configuration saved successfully")

    @classmethod
    def set(cls, config: "Configuration") -> None:
        """Set the singleton config instance directly, thread-safe."""
        log.debug("Setting new configuration instance")
        with cls._set_lock:
            global cfg
            cfg = config
        log.info("Configuration instance updated successfully")

    def _validate(self) -> None:
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

    @classmethod
    def _from_json(cls, json_path: str) -> "Configuration":
        """Loads a configuration from a JSON file."""

        def _resolve_field_type(field_def):
            """Resolve the real type for a dataclass field, handling Optional and generics (Python 3.12+)."""
            t = field_def.type
            origin = get_origin(t)
            if origin is Union:
                args = get_args(t)
                non_none = [a for a in args if a is not type(None)]
                if non_none:
                    return non_none[0]
            return t

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

        # Automatically instantiate all dataclass fields from JSON
        section_configs = {}
        for field_name, field_def in cls.__dataclass_fields__.items():
            field_type = _resolve_field_type(field_def)
            if field_name in config_params:
                if isinstance(field_type, type) and dataclasses.is_dataclass(field_type):
                    section_configs[field_name] = field_type(**config_params[field_name])
                else:
                    section_configs[field_name] = config_params[field_name]
            else:
                log.warning(f"Config field '{field_name}' not found in config file. Using default value.")
        # If a field is missing from the config file, dataclasses will use the default_factory/default value.
        log.info("Configuration loaded successfully.")
        return cls(**section_configs)

    def _to_json(self, json_path: str) -> None:
        """Saves the configuration to a JSON file."""
        log.info(f"Saving configuration to {json_path}...")
        json_file = Path(json_path)
        json_file.parent.mkdir(parents=True, exist_ok=True)

        config_dict = dataclasses.asdict(self)
        organized_config = {}

        for k, v in config_dict.items():
            if hasattr(v, "__dataclass_fields__"):
                organized_config[k] = dataclasses.asdict(v)
            else:
                organized_config[k] = v

        final_config = {"CuMind": organized_config}

        # Dump to string first
        json_str = json.dumps(final_config, indent=2)
        # Compact single-element lists: [\n  4\n] -> [4]
        json_str = re.sub(r"\[\s*([\d.eE+-]+)\s*\]", r"[\1]", json_str)

        with open(json_file, "w", encoding="utf-8") as f:
            f.write(json_str)
            f.write("\n")

        log.info("Configuration saved successfully.")


# Singleton instance for configuration
cfg: Configuration = Configuration()
