"""Global configuration management for CuMind."""

import dataclasses
import inspect
import json
import re
import subprocess
import threading
from pathlib import Path
from typing import Any, Optional, Tuple, Type, Union, cast, get_args, get_origin
from typing import Dict as DictType

from flax import nnx

from cumind.utils.resolve import resolve


@dataclasses.dataclass(frozen=True)
class HotSwappableConfig:
    """Base class for hot-swappable module configurations."""

    type: Optional[Union[str, Type[Any]]] = None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
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

        return cls(*args, **filtered_params, **kwargs)

    def extras(self) -> DictType[str, Any]:
        """Override this in subclasses to add global cfg values."""
        return {}


@dataclasses.dataclass(frozen=True)
class GeneralNetworksConfig:
    """Configuration for general networks."""

    hidden_state_dim: int = 128


@dataclasses.dataclass(frozen=True)
class RepresentationConfig(HotSwappableConfig):
    """Configuration for representation network."""

    type: Optional[Union[str, Type[Any]]] = "cumind.core.resnet.ResNet"
    num_hidden_layers: int = 2
    conv_channels: int = 32
    seed: int = 42

    def extras(self) -> DictType[str, Any]:
        input_dim = cfg.env.observation_shape
        hidden_dim = cfg.networks.hidden_state_dim
        rngs = nnx.Rngs(params=self.seed)
        return locals()


@dataclasses.dataclass(frozen=True)
class DynamicsConfig(HotSwappableConfig):
    """Configuration for dynamics network."""

    type: Optional[Union[str, Type[Any]]] = "cumind.core.mlp.MLPWithEmbedding"
    hidden_dim: int = 128
    num_hidden_layers: int = 2
    seed: int = 42

    def extras(self) -> DictType[str, Any]:
        input_dim = cfg.networks.hidden_state_dim
        embedding_size = cfg.env.action_space_size
        rngs = nnx.Rngs(params=self.seed)
        return locals()


@dataclasses.dataclass(frozen=True)
class PredictionConfig(HotSwappableConfig):
    """Configuration for prediction network."""

    type: Optional[Union[str, Type[Any]]] = "cumind.core.mlp.MLPDual"
    hidden_dim: int = 128
    num_hidden_layers: int = 2
    seed: int = 42

    def extras(self) -> DictType[str, Any]:
        input_dim = cfg.networks.hidden_state_dim
        num_actions = cfg.env.action_space_size
        rngs = nnx.Rngs(params=self.seed)
        return locals()


@dataclasses.dataclass(frozen=True)
class MemoryConfig(HotSwappableConfig):
    """Configuration for memory buffer."""

    type: Optional[Union[str, Type[Any]]] = "cumind.data.memory.PrioritizedMemoryBuffer"
    capacity: int = 2000
    min_size: int = 200
    min_pct: float = 0.1
    alpha: float = 0.6
    epsilon: float = 1e-6
    beta: float = 0.4

    def extras(self) -> DictType[str, Any]:
        return {}


@dataclasses.dataclass(frozen=True)
class TrainingConfig:
    """Configuration for training."""

    optimizer: str = "optax.adamw"
    batch_size: int = 64
    num_batches: int = 1
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    target_update_frequency: int = 100
    checkpoint_interval: int = 50
    num_episodes: int = 2000
    train_frequency: int = 2
    debug: bool = False


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
    max_episode_steps: int = 500


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
class LoggerConfig:
    """Configuration for logger."""

    wandb: bool = False
    title: str = "spamEggs"
    tags: list[str] = dataclasses.field(default_factory=lambda: ["foo", "bar"])

    level: str = "INFO"
    console: bool = True
    timestamps: bool = True
    tqdm: bool = False


@dataclasses.dataclass(frozen=True)
class TracingConfig:
    """Configuration for tracing."""

    enabled: bool = True

    host_level: int = 1
    # 0: Disable host tracing
    # 1: Trace user-instrumented events (default)
    # 2: Add high-level program execution details
    # 3: Add verbose, low-level details

    python_level: int = 0
    # 0 disables Python function call tracing, 1 enables (default)

    tpu_trace_mode: Optional[str] = "TRACE_ONLY_XLA"
    # TRACE_ONLY_HOST: Only host-side (CPU) activities are traced, no device traces.
    # TRACE_ONLY_XLA: Only XLA-level operations on the device are traced.
    # TRACE_COMPUTE: Traces compute operations on the device.
    # TRACE_COMPUTE_AND_SYNC: Traces both compute operations and synchronization events on the device.

    tpu_num_sparse_cores_to_trace: Optional[int] = 8
    # 8 sparse cores per chip is typical for TPUv4.

    tpu_num_sparse_core_tiles_to_trace: Optional[int] = 2
    # 2 tiles per sparse core is a sensible default.

    tpu_num_chips_to_profile_per_task: Optional[int] = 8
    # 8 chips for a full TPUv4-16 slice (topology 2x2x2).

    save_memory: bool = False


class ConfigMeta(type):
    def __getattribute__(cls, name: str) -> Any:
        # Allow access to dunder attributes
        if name.startswith("__") and name.endswith("__"):
            return super().__getattribute__(name)
        # Allow direct access to class attributes/methods to avoid recursion
        if name in ("_get_instance", "_instance", "_lock", "load", "save", "validate", "__dataclass_fields__"):
            return super().__getattribute__(name)
        # Redirect all other attribute access to the singleton instance
        instance = super().__getattribute__("_get_instance")()
        if hasattr(instance, name):
            return getattr(instance, name)
        raise AttributeError(f"Config has no field '{name}' (available fields: {list(instance.__dataclass_fields__.keys())})")


@dataclasses.dataclass(frozen=True)
class Configuration(metaclass=ConfigMeta):
    _instance: Optional["Configuration"] = None
    _lock = threading.RLock()
    """Main configuration for CuMind."""

    # Global nn settings
    networks: GeneralNetworksConfig = dataclasses.field(default_factory=GeneralNetworksConfig)

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
    logging: LoggerConfig = dataclasses.field(default_factory=LoggerConfig)
    tracing: TracingConfig = dataclasses.field(default_factory=TracingConfig)

    # Global settings
    device: str = "cpu"
    seed: int = 42
    multi_device: bool = False
    workspace: str = "artifacts"
    validate: bool = True

    @classmethod
    def _get_instance(cls) -> "Configuration":
        """Returns the singleton Configuration instance (thread-safe)."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def override(self, field: str, value: Any) -> "Configuration":
        """Override the value of a nested configuration field specified by a dotted path."""
        section, attr = field.split(".")
        target = self
        if hasattr(target, section):
            target = getattr(target, section)
        else:
            raise AttributeError(f"Config has no field '{section}' in path '{field}'")

        if hasattr(target, attr):
            object.__setattr__(target, attr, value)
        else:
            raise AttributeError(f"Config has no field '{attr}' in path '{field}'")
        return self

    def boot(self) -> Path:
        """Bootstraps experiment configuration and workspace, initializes logging, saves config, returns workspace path."""
        # Phase 1: Validate configuration
        self._validate()

        from pathlib import Path

        from cumind.utils.names import get_name

        # Phase 2: Prepare experiment directory
        for attempt in range(10):
            workspace_path = Path(self.workspace) / self.env.name / get_name()
            if not workspace_path.exists():
                try:
                    workspace_path.mkdir(parents=True, exist_ok=False)
                except FileExistsError:
                    continue
                break
        else:
            raise RuntimeError(f"Workspace directory already exists after 10 attempts: {workspace_path}")

        # Phase 3: Initialize logging and PRNG
        from cumind.utils.logger import log
        from cumind.utils.prng import key

        log().boot(self, workspace_path)
        key.seed(self.seed)

        config_json = self._as_json_str()
        log.info("Config values:\n" + config_json)

        # Phase 5: Save configuration copy to run directory
        config_copy_path = workspace_path / "configuration.json"
        self.save(config_copy_path)

        # Phase 6: Finalize boot process
        log.info(f"Experiment directory: {workspace_path}")
        log.info("Configuration booted successfully.")

        # Phase 7: Register Signals
        from cumind.utils.signals import register_signals

        register_signals()

        # Phase 8: Register XLA Flags
        from cumind.utils.xla_flags import register_xla_flags

        register_xla_flags()

        # Phase 9: Setup JAX profiler options if tracing is enabled
        from cumind.utils.tracing import start_trace

        start_trace(workspace_path)

        return workspace_path

    @classmethod
    def load(cls, obj: Optional[Union[str, Path, "Configuration"]] = None) -> Path:
        """Load configuration from a JSON file and set as singleton instance. Automatically validates after loading."""
        with cls._lock:
            if isinstance(obj, (str, Path)):
                cls._instance = cls._from_json(str(obj))
            elif isinstance(obj, Configuration):
                cls._instance = obj
            elif obj is None:
                cls._instance = Configuration()
            return cls._instance.boot()

    @classmethod
    def save(cls, path: Union[str, Path]) -> None:
        """Save configuration to a JSON file."""
        if cls._instance is None:
            cfg._instance = Configuration()
        cls._get_instance()._to_json(str(path))

    def _validate(self) -> None:
        """Validates the configuration parameters."""
        if not self.validate:
            return

        # GeneralNetworksConfig
        if self.networks.hidden_state_dim <= 0:
            raise ValueError(f"networks.hidden_dim must be positive, got {self.networks.hidden_state_dim}")

        # RepresentationConfig
        if self.representation.num_hidden_layers <= 0:
            raise ValueError(f"representation.num_hidden_layers must be positive, got {self.representation.num_hidden_layers}")
        if self.representation.conv_channels <= 0:
            raise ValueError(f"representation.conv_channels must be positive, got {self.representation.conv_channels}")
        if self.representation.type is None:
            raise ValueError("representation.type must be specified")

        # DynamicsConfig
        if self.dynamics.num_hidden_layers <= 0:
            raise ValueError(f"dynamics.num_hidden_layers must be positive, got {self.dynamics.num_hidden_layers}")
        if self.dynamics.type is None:
            raise ValueError("dynamics.type must be specified")

        # PredictionConfig
        if self.prediction.type is None:
            raise ValueError("prediction.type must be specified")

        # MemoryConfig
        if self.memory.capacity <= 0:
            raise ValueError(f"memory.capacity must be positive, got {self.memory.capacity}")
        if self.memory.min_size <= 0:
            raise ValueError(f"memory.min_size must be positive, got {self.memory.min_size}")
        if not (0 < self.memory.min_pct < 1):
            raise ValueError(f"memory.min_pct must be between 0 and 1, got {self.memory.min_pct}")
        if self.memory.alpha <= 0:
            raise ValueError(f"memory.per_alpha must be positive, got {self.memory.alpha}")
        if self.memory.epsilon <= 0:
            raise ValueError(f"memory.per_epsilon must be positive, got {self.memory.epsilon}")
        if self.memory.beta <= 0:
            raise ValueError(f"memory.per_beta must be positive, got {self.memory.beta}")
        if self.memory.type is None:
            raise ValueError("memory.type must be specified")

        # TrainingConfig
        if self.training.batch_size <= 0:
            raise ValueError(f"training.batch_size must be positive, got {self.training.batch_size}")
        if self.training.learning_rate <= 0:
            raise ValueError(f"training.learning_rate must be positive, got {self.training.learning_rate}")
        if self.training.weight_decay < 0:
            raise ValueError(f"training.weight_decay must be non-negative, got {self.training.weight_decay}")
        if self.training.target_update_frequency <= 0:
            raise ValueError(f"training.target_update_frequency must be positive, got {self.training.target_update_frequency}")
        if self.training.checkpoint_interval <= 0:
            raise ValueError(f"training.checkpoint_interval must be positive, got {self.training.checkpoint_interval}")
        if self.training.num_episodes <= 0:
            raise ValueError(f"training.num_episodes must be positive, got {self.training.num_episodes}")
        if self.training.train_frequency <= 0:
            raise ValueError(f"training.train_frequency must be positive, got {self.training.train_frequency}")
        if not isinstance(self.training.optimizer, str) or not self.training.optimizer:
            raise ValueError("training.optimizer must be a non-empty string")

        # MCTSConfig
        if self.mcts.num_simulations <= 0:
            raise ValueError(f"mcts.num_simulations must be positive, got {self.mcts.num_simulations}")
        if self.mcts.c_puct <= 0:
            raise ValueError(f"mcts.c_puct must be positive, got {self.mcts.c_puct}")
        if not (0 < self.mcts.dirichlet_alpha < 1):
            raise ValueError(f"mcts.dirichlet_alpha must be between 0 and 1, got {self.mcts.dirichlet_alpha}")
        if not (0 < self.mcts.exploration_fraction < 1):
            raise ValueError(f"mcts.exploration_fraction must be between 0 and 1, got {self.mcts.exploration_fraction}")

        # EnvironmentConfig
        if not isinstance(self.env.name, str) or not self.env.name:
            raise ValueError("env.name must be a non-empty string")
        if self.env.action_space_size <= 0:
            raise ValueError(f"env.action_space_size must be positive, got {self.env.action_space_size}")
        if not isinstance(self.env.observation_shape, tuple) or not all(isinstance(x, int) and x > 0 for x in self.env.observation_shape):
            raise ValueError("env.observation_shape must be a tuple of positive integers")
        if not isinstance(self.env.max_episode_steps, int) or self.env.max_episode_steps <= 0:
            raise ValueError(f"env.max_episode_steps must be a positive integer, got {self.env.max_episode_steps}")

        # SelfPlayConfig
        if self.selfplay.num_unroll_steps <= 0:
            raise ValueError(f"selfplay.num_unroll_steps must be positive, got {self.selfplay.num_unroll_steps}")
        if self.selfplay.td_steps <= 0:
            raise ValueError(f"selfplay.td_steps must be positive, got {self.selfplay.td_steps}")
        if not (0 < self.selfplay.discount < 1):
            raise ValueError(f"selfplay.discount must be between 0 and 1, got {self.selfplay.discount}")

        # DataTypesConfig
        valid_model_dtypes = ["float32", "float16", "bfloat16"]
        if self.dtypes.model not in valid_model_dtypes:
            raise ValueError(f"dtypes.model must be one of {valid_model_dtypes}, got {self.dtypes.model}")
        valid_action_dtypes = ["int32", "int64"]
        if self.dtypes.action not in valid_action_dtypes:
            raise ValueError(f"dtypes.action must be one of {valid_action_dtypes}, got {self.dtypes.action}")
        valid_target_dtypes = ["float32", "float16", "bfloat16"]
        if self.dtypes.target not in valid_target_dtypes:
            raise ValueError(f"dtypes.target must be one of {valid_target_dtypes}, got {self.dtypes.target}")

        # LoggerConfig
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.logging.level not in valid_levels:
            raise ValueError(f"logging.level must be one of {valid_levels}, got {self.logging.level}")
        if not isinstance(self.logging.console, bool):
            raise ValueError(f"logging.console must be a boolean, got {type(self.logging.console)}")
        if not isinstance(self.logging.timestamps, bool):
            raise ValueError(f"logging.timestamps must be a boolean, got {type(self.logging.timestamps)}")
        if self.logging.title is not None and not isinstance(self.logging.title, str):
            raise ValueError(f"logging.wandb_name must be a string, got {type(self.logging.title)}")
        if not isinstance(self.logging.tags, list) or not all(isinstance(tag, str) for tag in self.logging.tags):
            raise ValueError("logging.tags must be a list of strings")
        # experiment_dir
        if not isinstance(self.workspace, str) or not self.workspace:
            raise ValueError("experiment_dir must be a non-empty string")

        # TracingConfig
        if not isinstance(self.tracing.enabled, bool):
            raise ValueError(f"tracing.enabled must be a boolean, got {type(self.tracing.enabled)}")
        if not isinstance(self.tracing.host_level, int) or self.tracing.host_level < 0:
            raise ValueError(f"tracing.host_level must be a non-negative integer, got {self.tracing.host_level}")
        if not isinstance(self.tracing.python_level, int) or self.tracing.python_level < 0:
            raise ValueError(f"tracing.python_level must be a non-negative integer, got {self.tracing.python_level}")
        if not isinstance(self.tracing.save_memory, bool):
            raise ValueError(f"tracing.save_memory must be a boolean, got {type(self.tracing.save_memory)}")

        # device
        valid_devices = ["cpu", "cuda", "tpu", "rocm", "metal"]
        if self.device not in valid_devices:
            raise ValueError(f"device must be one of {valid_devices}, got {self.device}")
        if not isinstance(self.multi_device, bool):
            raise ValueError(f"multi_device must be a boolean, got {type(self.multi_device)}")
        if self.multi_device and self.device == "cpu":
            raise ValueError("multi_device cannot be True when device is 'cpu'")

        # seed
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError(f"seed must be a non-negative integer, got {self.seed}")

        # ----------------------------------------------------------
        # Dynamic backend resolution.
        # This block installs the appropriate JAX backend depending
        # on the selected device. It uses 'uv' for pip installation.
        # For multi-device setups (TPU/CUDA), it initializes JAX
        # distributed system.
        # ----------------------------------------------------------

        if self.device == "cuda":
            subprocess.run(["uv", "pip", "install", "jax[cuda12]>=0.6.2"], check=True)
        elif self.device == "tpu":
            subprocess.run(["uv", "pip", "install", "jax[tpu]>=0.6.2"], check=True)
        elif self.device == "rocm":
            subprocess.run(["uv", "pip", "install", "jax[rocm]>=0.6.2"], check=True)
        elif self.device == "metal":
            subprocess.run(["uv", "pip", "install", "jax[metal]>=0.6.2"], check=True)

        # Initialize JAX distributed system for TPU and multi-device setups
        if (self.device == "tpu" or self.device == "cuda") and self.multi_device:
            import jax

            try:
                jax.distributed.initialize()
            except Exception as e:
                raise RuntimeError(f"JAX distributed initialization failed: {e}. Continuing with single-process mode.")

    @classmethod
    def _from_json(cls, json_path: str) -> "Configuration":
        """Loads a configuration from a JSON file."""

        def _resolve_field_type(field_def: dataclasses.Field[Any]) -> type[Any]:
            """Resolve the real type for a dataclass field, handling Optional and generics (Python 3.12+)."""
            t = field_def.type
            origin = get_origin(t)
            if origin is Union:
                args = get_args(t)
                non_none = [a for a in args if a is not type(None)]
                if non_none:
                    return cast(type[Any], non_none[0])
            return cast(type[Any], t)

        def _coerce_types(data: dict[str, Any], dataclass_type: type[Any]) -> dict[str, Any]:
            """Recursively coerce types in a dict to match the dataclass field types."""
            result: dict[str, Any] = {}
            for field_name, field_def in dataclass_type.__dataclass_fields__.items():
                if field_name not in data:
                    continue
                value = data[field_name]
                field_type = _resolve_field_type(field_def)
                origin = get_origin(field_type)
                # Nested dataclass
                if isinstance(field_type, type) and dataclasses.is_dataclass(field_type) and isinstance(value, dict):
                    result[field_name] = field_type(**_coerce_types(value, field_type))
                # Tuple[...] field
                elif origin is tuple and isinstance(value, list):
                    result[field_name] = tuple(value)
                else:
                    result[field_name] = value
            return result

        json_file = Path(json_path)
        if not json_file.exists():
            raise FileNotFoundError(f"Config file not found: {json_path}")

        with open(json_file, "r", encoding="utf-8") as f:
            config_dict = json.load(f)

        if "CuMind" not in config_dict:
            raise ValueError("Root key 'CuMind' not found in config file.")

        config_params = config_dict["CuMind"]

        section_configs = _coerce_types(config_params, cls)
        return cls(**section_configs)

    def _as_json_str(self) -> str:
        """Returns the configuration as a JSON string."""
        config_dict = dataclasses.asdict(self)
        organized_config = {}
        for k, v in config_dict.items():
            if k.startswith("_"):
                continue
            if hasattr(v, "__dataclass_fields__"):
                organized_config[k] = dataclasses.asdict(v)
            else:
                organized_config[k] = v
        final_config = {"CuMind": organized_config}
        json_str = json.dumps(final_config, indent=2)
        json_str = re.sub(r"\[\s*\n\s*(.*?)\s*\n\s*\]", lambda m: "[" + re.sub(r"\s*\n\s*", " ", m.group(1).strip()) + "]", json_str, flags=re.DOTALL)
        return json_str

    def _to_json(self, json_path: str) -> None:
        """Saves the configuration to a JSON file."""
        json_file = Path(json_path)
        json_file.parent.mkdir(parents=True, exist_ok=True)
        json_str = self._as_json_str()
        with open(json_file, "w", encoding="utf-8") as f:
            f.write(json_str)
            f.write("\n")


# Alias
cfg = Configuration
