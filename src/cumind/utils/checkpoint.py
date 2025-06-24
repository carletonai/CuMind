"""Checkpointing utilities for saving and loading model and training states."""

import pickle
from pathlib import Path
from typing import Any, Dict, cast

from flax import nnx


def save_checkpoint(state: Dict[str, Any], path: str) -> None:
    """Save training checkpoint to a file.

    Args:
        state: A dictionary containing the state to save (e.g., network, optimizer).
        path: File path to save the checkpoint.
    """
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with open(path_obj, "wb") as f:
        pickle.dump(state, f)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(path: str) -> Dict[str, Any]:
    """Load training checkpoint from a file.

    Args:
        path: File path to load the checkpoint from.

    Returns:
        The loaded state dictionary.
    """
    with open(path, "rb") as f:
        state = pickle.load(f)
    print(f"Checkpoint loaded from {path}")
    return cast(Dict[str, Any], state)
