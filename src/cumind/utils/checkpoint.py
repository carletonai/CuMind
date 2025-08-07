"""Checkpointing utilities for saving and loading model and training states."""

import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict

from cumind.utils.logger import log


class CheckpointMetadata(TypedDict, total=False):
    """Metadata stored in checkpoints."""

    episode: int
    train_step_count: int
    last_loss: Dict[str, float]
    memory_size: int


class AgentState(TypedDict):
    """Agent state structure for checkpoints."""

    network_state: Any
    optimizer_state: Any
    memory_state: Optional[Any]


class CheckpointData(TypedDict):
    """Complete checkpoint data structure."""

    state: AgentState
    metadata: CheckpointMetadata
    timestamp: str


def save_checkpoint(state: AgentState, path: str, metadata: Optional[CheckpointMetadata] = None) -> None:
    """Save training checkpoint to a file.
    Args:
        state: The agent state dictionary to save.
        path: The file path to save the checkpoint.
        metadata: Optional metadata to include in the checkpoint.
    """
    checkpoint_data: CheckpointData = {"state": state, "metadata": metadata or {}, "timestamp": datetime.now().isoformat()}

    os.makedirs(os.path.dirname(path), exist_ok=True)

    try:
        with open(path, "wb") as f:
            pickle.dump(checkpoint_data, f)
        log.info(f"Checkpoint saved to {path}")
    except Exception as e:
        log.exception(f"Failed to save checkpoint to {path}: {e}")
        raise


def load_checkpoint(path: str) -> CheckpointData:
    """Load training checkpoint from a file.

    Args:
        path: File path to load the checkpoint from.

    Returns:
        The full checkpoint data.
    """
    try:
        with open(path, "rb") as f:
            checkpoint_data: CheckpointData = pickle.load(f)
        log.info(f"Checkpoint loaded from {path}")

        return checkpoint_data
    except Exception as e:
        log.exception(f"Failed to load checkpoint from {path}: {e}")
        raise


def get_checkpoint_files(checkpoint_dir: str) -> List[Path]:
    """Returns a sorted list of checkpoint files in a directory."""
    checkpoint_path = Path(checkpoint_dir)
    return sorted(checkpoint_path.glob("*.pkl"))


def find_latest_checkpoint_in_dir(checkpoint_dir: str) -> Optional[str]:
    """Finds the latest checkpoint file in a directory."""
    files = get_checkpoint_files(checkpoint_dir)
    if not files:
        return None
    latest_file = max(files, key=lambda f: f.stat().st_mtime)
    return str(latest_file)


def latest_checkpoints(root_dir: str) -> Dict[str, List[Tuple[str, datetime]]]:
    """Scans for available checkpoints and returns a structured dictionary."""
    root_path = Path(root_dir)
    checkpoints: Dict[str, List[Tuple[str, datetime]]] = {}
    if not root_path.is_dir():
        return checkpoints

    for env_dir in root_path.iterdir():
        if env_dir.is_dir():
            env_name = env_dir.name
            runs = []
            for run_dir in env_dir.iterdir():
                if run_dir.is_dir():
                    try:
                        timestamp = datetime.strptime(run_dir.name, "%Y%m%d-%H%M%S")
                        latest_checkpoint = find_latest_checkpoint_in_dir(str(run_dir))
                        if latest_checkpoint:
                            runs.append((latest_checkpoint, timestamp))
                    except ValueError:
                        # Ignore directories that don't match the timestamp format
                        continue
            if runs:
                # Sort by timestamp, descending
                runs.sort(key=lambda x: x[1], reverse=True)
                checkpoints[env_name] = runs
    return checkpoints


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

    # Find all subdirectories
    timestamp_dirs = [d for d in checkpoint_dir.iterdir() if d.is_dir()]
    if not timestamp_dirs:
        log.warning(f"No checkpoint directories found in {checkpoint_dir}.")
        return None

    # Get the latest directory
    latest_timestamp_dir = sorted(timestamp_dirs)[-1]

    # Find checkpoint files in the latest directory
    checkpoint_files = sorted(latest_timestamp_dir.glob("*.pkl"))
    if not checkpoint_files:
        log.warning(f"No checkpoint files found in {latest_timestamp_dir}.")
        return None

    latest_checkpoint = str(checkpoint_files[-1])
    log.info(f"Found latest checkpoint: {latest_checkpoint}")
    return latest_checkpoint
