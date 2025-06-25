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
        level: int = logging.DEBUG,
    ):
        """Initialize logger with optional TensorBoard and W&B support.

        Args:
            log_dir: Directory to store log files
            wandb_project: Weights & Biases project name (optional)
            wandb_config: Configuration dict for W&B (optional)
            level: Logging level (default: logging.INFO)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Setup file logging
        self.logger = logging.getLogger("CuMind")
        self.logger.setLevel(level)

        handler = logging.FileHandler(self.log_dir / "training.log")
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        # TensorBoard writer
        self.tensorboard_writer = None
        '''
        try:
            from tensorboard import SummaryWriter  # type: ignore

            self.tensorboard_writer = SummaryWriter(log_dir=str(self.log_dir))
        except ImportError:
            self.logger.info("TensorBoard not available, skipping TensorBoard logging")
        '''
        # W&B setup
        self.use_wandb = wandb_project is not None
        if self.use_wandb:
            wandb.init(project=wandb_project, config=wandb_config, dir=str(self.log_dir))

    def log_scalar(self, name: str, value: float, step: int) -> None:
        """Log a scalar value.

        Args:
            name: Name of the metric
            value: Scalar value to log
            step: Training step/epoch
        """
        self.logger.info("Step %d: %s = %.6f", step, name, value)

        if self.tensorboard_writer is not None:
            self.tensorboard_writer.add_scalar(name, value, step)

        if self.use_wandb:
            wandb.log({name: value}, step=step)

    def log_scalars(self, metrics: Dict[str, float], step: int) -> None:
        """Log multiple scalar values.

        Args:
            metrics: Dictionary of metric names and values
            step: Training step/epoch
        """
        for name, value in metrics.items():
            self.log_scalar(name, value, step)

    def log(self, level: int, msg: str, *args, **kwargs) -> None:
        """Log a message at a specific logging level."""
        self.logger.log(level, msg, *args, **kwargs)
        if self.use_wandb:
            wandb.log({logging.getLevelName(level).lower(): msg % args if args else msg})

    def log_critical(self, text: str) -> None:
        """Log critical message."""
        self.logger.critical(text)
        if self.use_wandb:
            wandb.log({"critical": text})

    def log_fatal(self, text: str) -> None:
        """Log fatal message (alias for critical)."""
        self.logger.fatal(text)
        if self.use_wandb:
            wandb.log({"fatal": text})

    def log_error(self, text: str) -> None:
        """Log error message."""
        self.logger.error(text)
        if self.use_wandb:
            wandb.log({"error": text})

    def log_exception(self, text: str) -> None:
        """Log exception message (with traceback)."""
        self.logger.exception(text)
        if self.use_wandb:
            wandb.log({"exception": text})

    def log_warning(self, text: str) -> None:
        """Log warning message."""
        self.logger.warning(text)
        if self.use_wandb:
            wandb.log({"warning": text})

    def log_warn(self, text: str) -> None:
        """Log warn message (alias for warning)."""
        self.logger.warn(text)
        if self.use_wandb:
            wandb.log({"warn": text})

    def log_info(self, text: str) -> None:
        """Log info message."""
        self.logger.info(text)
        if self.use_wandb:
            wandb.log({"info": text})

    def log_debug(self, text: str) -> None:
        """Log debug message."""
        self.logger.debug(text)
        if self.use_wandb:
            wandb.log({"debug": text})

    def close(self) -> None:
        """Close logger and cleanup resources."""
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.close()

        if self.use_wandb:
            wandb.finish()

        # Close logging handlers
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)
