"""Checkpointing utilities for saving and loading model and training states."""

# AI slop code

import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

from cumind.utils.logger import log


def save_checkpoint(state: Dict[str, Any], path: Path) -> None:
    """Save training checkpoint to a file.

    Args:
        state: A dictionary containing the state to save (e.g., network, optimizer).
        path: File path to save the checkpoint.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
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
                    # Parse run directory modification time since we use human-readable names now
                    stat = run_dir.stat()
                    timestamp = datetime.fromtimestamp(stat.st_mtime)
                    latest_checkpoint = find_latest_checkpoint_in_dir(str(run_dir))
                    if latest_checkpoint:
                        runs.append((latest_checkpoint, timestamp))
            if runs:
                runs.sort(key=lambda x: x[1], reverse=True)
                checkpoints[env_name] = runs
    return checkpoints


def find_latest_checkpoint_for_env(env_name: str, root_dir: str = "artifacts") -> str | None:
    """Find the latest checkpoint for a specific environment.

    Args:
        env_name: Name of the environment to find checkpoints for.
        root_dir: Root experiments directory.

    Returns:
        Path to the latest checkpoint file, or None if not found.
    """
    env_dir = Path(root_dir) / env_name

    if not env_dir.is_dir():
        log.warning(f"Environment directory not found: {env_dir}")
        return None

    # Find all run directories
    run_dirs = [d for d in env_dir.iterdir() if d.is_dir()]
    if not run_dirs:
        log.warning(f"No run directories found in {env_dir}.")
        return None

    # Get the latest directory by modification time
    latest_run_dir = max(run_dirs, key=lambda d: d.stat().st_mtime)

    # Find checkpoint files in the latest directory
    checkpoint_files = sorted(latest_run_dir.glob("episode_*.pkl"))
    if not checkpoint_files:
        log.warning(f"No checkpoint files found in {latest_run_dir}.")
        return None

    latest_checkpoint = str(checkpoint_files[-1])
    log.info(f"Found latest checkpoint: {latest_checkpoint}")
    return latest_checkpoint


def create_latest_checkpoint_link(workspace: str, checkpoint_path: str) -> None:
    """Create or update a symlink to the latest checkpoint in the run directory."""
    import os

    workspace_path = Path(workspace)
    latest_link = workspace_path / "latest_checkpoint.pkl"

    # Remove existing symlink if it exists
    if latest_link.exists() or latest_link.is_symlink():
        latest_link.unlink()

    # Create relative symlink to the checkpoint file
    relative_path = os.path.relpath(checkpoint_path, workspace)
    latest_link.symlink_to(relative_path)
    log.info(f"Updated latest checkpoint link: {latest_link} -> {relative_path}")


def get_runs_for_env(env_name: str, root_dir: str = "experiments") -> List[Tuple[str, datetime, str]]:
    """Get all runs for a specific environment with human-readable timestamps.

    Returns:
        List of tuples (workspace_name, creation_time, workspace_path)
    """
    env_dir = Path(root_dir) / env_name
    if not env_dir.is_dir():
        return []

    runs = []
    for run_dir in env_dir.iterdir():
        if run_dir.is_dir():
            stat = run_dir.stat()
            creation_time = datetime.fromtimestamp(stat.st_ctime)
            runs.append((run_dir.name, creation_time, str(run_dir)))

    return sorted(runs, key=lambda x: x[1], reverse=True)


def get_all_environments(root_dir: str = "experiments") -> List[str]:
    """Get list of all environments that have experiment runs."""
    root_path = Path(root_dir)
    if not root_path.is_dir():
        return []

    return [env_dir.name for env_dir in root_path.iterdir() if env_dir.is_dir()]


def format_datetime_short(dt: datetime) -> str:
    """Format datetime in short, human-readable format like 'Aug 15th 5:24pm'."""
    day_suffix = "th"
    if dt.day % 10 == 1 and dt.day != 11:
        day_suffix = "st"
    elif dt.day % 10 == 2 and dt.day != 12:
        day_suffix = "nd"
    elif dt.day % 10 == 3 and dt.day != 13:
        day_suffix = "rd"

    month_abbr = dt.strftime("%b")
    day = f"{dt.day}{day_suffix}"
    time_str = dt.strftime("%I:%M%p").lower()

    return f"{month_abbr} {day} {time_str}"


def lookup_run_by_name(run_name: str, root_dir: str = "experiments") -> Optional[Tuple[str, str, datetime]]:
    """Find a run by its name across all environments.

    Returns:
        Tuple of (env_name, run_path, creation_time) or None if not found
    """
    root_path = Path(root_dir)
    if not root_path.is_dir():
        return None

    for env_dir in root_path.iterdir():
        if env_dir.is_dir():
            run_dir = env_dir / run_name
            if run_dir.is_dir():
                stat = run_dir.stat()
                creation_time = datetime.fromtimestamp(stat.st_ctime)
                return (env_dir.name, str(run_dir), creation_time)

    return None
