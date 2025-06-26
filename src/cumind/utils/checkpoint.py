"""Checkpointing utilities for saving and loading model and training states."""

import pickle
from pathlib import Path
from typing import Any, Dict, cast

from flax import nnx

from .logger import log


def save_checkpoint(state: Dict[str, Any], path: str) -> None:
    """Save training checkpoint to a file.

    Args:
        state: A dictionary containing the state to save (e.g., network, optimizer).
        path: File path to save the checkpoint.
    """
    try:
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        with open(path_obj, "wb") as f:
            pickle.dump(state, f)
        log.info(f"Checkpoint saved to {path}")
    except (IOError, pickle.PicklingError) as e:
        log.exception(f"Failed to save checkpoint to {path}: {e}")
        raise


def load_checkpoint(path: str) -> Dict[str, Any]:
    """Load training checkpoint from a file.

    Args:
        path: File path to load the checkpoint from.

    Returns:
        The loaded state dictionary.
    """
    try:
        with open(path, "rb") as f:
            state = pickle.load(f)
        log.info(f"Checkpoint loaded from {path}")
        return cast(Dict[str, Any], state)
    except (IOError, pickle.UnpicklingError, FileNotFoundError) as e:
        log.exception(f"Failed to load checkpoint from {path}: {e}")
        raise
