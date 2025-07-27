"""Dynamic import utilities for CuMind."""

import importlib
from typing import Any


def resolve(dotted: str) -> Any:
    """Import and return a class/function from a dotted string path."""
    module, attr = dotted.rsplit(".", 1)
    try:
        mod = importlib.import_module(module)
        return getattr(mod, attr)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Could not resolve '{dotted}': {e}")
