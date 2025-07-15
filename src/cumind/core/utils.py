from typing import Callable, Optional, Tuple, Union

from flax import nnx


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
