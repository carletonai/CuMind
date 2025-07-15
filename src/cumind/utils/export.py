"""Network instantiation and export utilities."""

import inspect
from typing import Any, Callable, Dict, Optional, Tuple, Union

from flax import nnx

from ..config import Config
from . import log
from .prng import key


def str_to_nnx(*names: Optional[Union[str, nnx.Module]]) -> Tuple[Callable[..., nnx.Module], ...]:
    """
    Return callables from the module's globals by their names.

    Raises:
        AttributeError: If a name is not found or not callable.
        ValueError: If no names are provided.
    """
    if not names:
        raise ValueError("At least one string name must be provided.")
    result = []
    for name in names:
        if not isinstance(name, str):
            raise ValueError("All names must be strings.")
        obj = globals().get(name)
        if obj is None:
            raise AttributeError(f"No object named '{name}' found.")
        if not callable(obj):
            raise AttributeError(f"Object '{name}' is not callable.")
        result.append(obj)
    if not result:
        raise ValueError("No valid callables found for the given names.")
    return tuple(result)


def instantiate_network(network_cls, network_type: str, cfg: Config) -> nnx.Module:
    """Instantiate a network using inspect.signature() for dynamic argument matching."""
    sig = inspect.signature(network_cls.__init__)

    available_args: Dict[str, Any] = {"rngs": nnx.Rngs(params=key())}

    if cfg.network_architecture:
        network_args = cfg.network_architecture.get(network_type, {})
        available_args.update(network_args)

        for param_name, value in cfg.network_architecture.items():
            if param_name not in ["representation_network", "dynamics_network", "prediction_network"]:
                available_args[param_name] = value

    if network_type == "representation_network":
        available_args["input_shape"] = cfg.observation_shape
    elif network_type == "dynamics_network":
        available_args["embedding_size"] = cfg.action_space_size
    elif network_type == "prediction_network":
        available_args["output_size"] = cfg.action_space_size

    final_args = {}
    for param_name in sig.parameters:
        if param_name == "self":
            continue
        if param_name in available_args:
            final_args[param_name] = available_args[param_name]

    try:
        bound_args = sig.bind(None, **final_args)
        bound_args.apply_defaults()
    except TypeError as e:
        log.critical(f"Failed to bind arguments for {network_type}: {e}")
        raise ValueError(f"Invalid arguments for {network_type}: {e}") from e

    instantiation_args = dict(bound_args.arguments)
    instantiation_args.pop("self", None)

    network_instance = network_cls(**instantiation_args)
    if not isinstance(network_instance, nnx.Module):
        raise ValueError(f"{network_type} must be nnx.Module, got {type(network_instance)}")

    log.debug(f"Successfully instantiated {network_type} with args: {instantiation_args}")
    return network_instance


def validate_networks(cfg: Config) -> None:
    """Validate network configurations by attempting to instantiate them."""
    network_configs = [
        (cfg.representation_network, "representation_network"),
        (cfg.dynamics_network, "dynamics_network"),
        (cfg.prediction_network, "prediction_network"),
    ]

    # assert strings
    for network_name, field_name in network_configs:
        if not isinstance(network_name, str):
            log.critical(f"Invalid {field_name}: {network_name}. Must be a string.")
            raise ValueError(f"{field_name} must be a string, got {type(network_name).__name__}")

    # extract nnx
    try:
        result = str_to_nnx(cfg.representation_network, cfg.dynamics_network, cfg.prediction_network)
        network_classes = list(result)
    except Exception as e:
        log.critical(f"Failed to get network classes: {e}")
        raise ValueError(f"Invalid network configuration: {e}") from e

    #instantiate nnx
    network_types = ["representation_network", "dynamics_network", "prediction_network"]
    for network_cls, network_type in zip(network_classes, network_types):
        try:
            instantiate_network(network_cls, network_type, cfg)
        except Exception as e:
            log.critical(f"Failed to instantiate {network_type}: {e}")
            raise ValueError(f"Network instantiation failed for {network_type}: {e}") from e

    log.debug("Network validation successful - all networks instantiated correctly")

