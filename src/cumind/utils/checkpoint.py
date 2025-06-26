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


def find_latest_checkpoint_in_dir(directory: str) -> str | None:
    """Find the latest checkpoint file in a specific directory.

    Args:
        directory: Path to the directory containing checkpoint files.

    Returns:
        Path to the latest checkpoint file, or None if not found.
    """
    checkpoint_dir = Path(directory)
    if not checkpoint_dir.is_dir():
        log.warning(f"Checkpoint directory not found: {directory}")
        return None

    checkpoint_files = sorted(checkpoint_dir.glob("*.pkl"))
    if not checkpoint_files:
        log.warning(f"No checkpoint files found in {directory}.")
        return None

    latest_checkpoint = str(checkpoint_files[-1])
    log.info(f"Found latest checkpoint: {latest_checkpoint}")
    return latest_checkpoint


def find_latest_checkpoint_for_env(env_name: str) -> str | None:
    """Find the latest checkpoint for a specific environment.

    Args:
        env_name: Name of the environment to find checkpoints for.

    Returns:
        Path to the latest checkpoint file, or None if not found.
    """
    checkpoint_dir = Path("checkpoints") / env_name

    if not checkpoint_dir.is_dir():
        log.warning(f"Checkpoint directory not found: {checkpoint_dir}")
        return None

    # Find all timestamp subdirectories
    timestamp_dirs = [d for d in checkpoint_dir.iterdir() if d.is_dir()]
    if not timestamp_dirs:
        log.warning(f"No checkpoint directories found in {checkpoint_dir}.")
        return None

    # Get the latest timestamp directory
    latest_timestamp_dir = sorted(timestamp_dirs)[-1]

    # Find checkpoint files in the latest timestamp directory
    checkpoint_files = sorted(latest_timestamp_dir.glob("*.pkl"))
    if not checkpoint_files:
        log.warning(f"No checkpoint files found in {latest_timestamp_dir}.")
        return None

    latest_checkpoint = str(checkpoint_files[-1])
    log.info(f"Found latest checkpoint: {latest_checkpoint}")
    return latest_checkpoint
