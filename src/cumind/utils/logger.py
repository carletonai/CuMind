"""Unified logger with TensorBoard and Weights & Biases support."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import wandb


class Logger:
    """Unified logger for training metrics and visualizations."""

    def __init__(
        self,
        log_dir: str = "logs",
        wandb_project: Optional[str] = None,
        wandb_config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize logger with optional TensorBoard and W&B support.

        Args:
            log_dir: Directory to store log files
            wandb_project: Weights & Biases project name (optional)
            wandb_config: Configuration dict for W&B (optional)

        Implementation:
            - Setup file logging to log_dir
            - Initialize TensorBoard writer
            - Setup W&B if project name provided
            - Store configuration parameters
        """
        # Branch: feature/logger-init
        raise NotImplementedError("Logger.__init__ needs to be implemented")

    def log_scalar(self, name: str, value: float, step: int) -> None:
        """Log a scalar value.

        Args:
            name: Name of the metric
            value: Scalar value to log
            step: Training step/epoch

        Implementation:
            - Log to file logger
            - Send to TensorBoard if available
            - Send to W&B if configured
        """
        # Branch: feature/log-scalar
        raise NotImplementedError("Logger.log_scalar needs to be implemented")

    def log_scalars(self, metrics: Dict[str, float], step: int) -> None:
        """Log multiple scalar values.

        Args:
            metrics: Dictionary of metric names and values
            step: Training step/epoch

        Implementation:
            - Iterate through metrics dictionary
            - Call log_scalar for each metric
        """
        # Branch: feature/log-scalars
        raise NotImplementedError("Logger.log_scalars needs to be implemented")

    def log_text(self, text: str) -> None:
        """Log text message.

        Args:
            text: Text message to log

        Implementation:
            - Log to file logger with timestamp
            - Send to W&B if configured
        """
        # Branch: feature/log-text
        raise NotImplementedError("Logger.log_text needs to be implemented")

    def close(self) -> None:
        """Close logger and cleanup resources.

        Implementation:
            - Close file handlers
            - Close TensorBoard writer if active
            - Finish W&B run if active
        """
        # Branch: feature/logger-close
        raise NotImplementedError("Logger.close needs to be implemented")
