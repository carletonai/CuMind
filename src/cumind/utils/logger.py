"""Unified logger with TensorBoard and Weights & Biases support."""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# import tensorboard as tb
# import wandb


def get_logger():
    """Get the logger instance.
    Usage:
        from cumind.utils.logger import get_logger
        logger = get_logger()
        logger.info("hello world")
        logger.info("this is how you use this")
    """
    return Logger()


class _LogFunctor:
    """A functor to provide direct access to the get_logger() instance's methods,
    but restricts to only known logging methods.
    Usage:
        from cumind.utils.logger import log
        log.info("hello world")
        log.info("this is how you use this")
    """

    _allowed_methods = {
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "critical",
        "log",
        "log_scalar",
        "log_scalars",
        "close",
    }

    def __getattr__(self, name):
        if name not in self._allowed_methods:
            raise AttributeError(f"'log' object has no attribute '{name}'")
        return getattr(get_logger(), name)


log = _LogFunctor()


class Logger:
    """A wrapper around the standard Python logger to provide a unified, configurable interface."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        log_dir: str = "logs",
        level: str = "INFO",
        wandb_config: Optional[Dict[str, Any]] = None,
        tensorboard_config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize and configure the logger.

        Args:
            log_dir: Directory to store log files.
            level: Logging level.
            wandb_config: Configuration dict for W&B (optional).
            tensorboard_config: Configuration dict for TensorBoard (optional).
        """
        # Only initialize once
        if hasattr(self, "_logger"):
            return

        self._logger = logging.getLogger("CuMindLogger")
        self._logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.use_wandb = wandb_config is not None  # wandb.run
        self.use_tensorboard = tensorboard_config is not None

        # Prevent adding handlers multiple times
        if not self._logger.handlers:
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%I:%M:%S %p")

            file_handler = logging.FileHandler(self.log_dir / "training.log")
            file_handler.setFormatter(formatter)
            self._logger.addHandler(file_handler)

        # Initialize W&B
        if self.use_wandb:
            raise NotImplementedError("Weights & Biases is not implemented yet.")

        # Initialize TensorBoard
        if self.use_tensorboard:
            raise NotImplementedError("TensorBoard is not implemented yet.")

    def debug(self, msg: str, *args, **kwargs):
        """Logs a message with level DEBUG."""
        self._logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs):
        """Logs a message with level INFO."""
        self._logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        """Logs a message with level WARNING."""
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        """Logs a message with level ERROR."""
        self._logger.error(msg, *args, **kwargs)

    def exception(self, msg: str, *args, **kwargs):
        """Logs a message with level ERROR, including exception info."""
        self._logger.exception(msg, *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs):
        """Logs a message with level CRITICAL."""
        self._logger.critical(msg, *args, **kwargs)

    def log(self, level: int, msg: str, *args, **kwargs):
        """Logs a message with the specified level."""
        self._logger.log(level, msg, *args, **kwargs)

    def log_scalar(self, name: str, value: float, step: int) -> None:
        """Log a scalar value to the console and to W&B."""
        self.info(f"Step {step:4d}: {name} = {value:.6f}")
        if self.use_wandb:
            raise NotImplementedError("Weights & Biases is not implemented yet.")
        if self.use_tensorboard:
            raise NotImplementedError("TensorBoard is not implemented yet.")

    def log_scalars(self, metrics: Dict[str, float], step: int) -> None:
        """Log multiple scalar values."""
        for name, value in metrics.items():
            self.log_scalar(name, value, step)

    def close(self) -> None:
        """Close logger and cleanup resources, like file handlers and W&B run."""
        if self.use_wandb:
            raise NotImplementedError("Weights & Biases is not implemented yet.")
        if self.use_tensorboard:
            raise NotImplementedError("TensorBoard is not implemented yet.")

        for handler in self._logger.handlers[:]:
            handler.close()
            self._logger.removeHandler(handler)

        self._logger.info("Logger is cleaning up resources.")
        logging.shutdown()
