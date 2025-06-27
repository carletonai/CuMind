"""Unified logger with TensorBoard and Weights & Biases support."""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional, cast

import tensorboard as tb  # type: ignore
import wandb


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
        "set_level",
    }

    def __getattr__(self, name: str) -> Callable[..., None]:
        if name not in self._allowed_methods:
            raise AttributeError(f"'log' object has no attribute '{name}'")
        return cast(Callable[..., None], getattr(Logger(), name))


log = _LogFunctor()


class Logger:
    """A singleton logger that provides a unified, configurable interface.

    The Logger is implemented as a singleton, meaning that it is configured
    only once when the first instance is created. Subsequent calls to the
    constructor will return the already-existing instance without
    re-initializing it. This ensures a single, consistent logging setup
    throughout the application.

    The primary way to use the logger is through the `log` object, which
    provides a direct, convenient interface.

    Usage:
        from cumind.utils.logger import log

        # Use the `log` object directly - no setup required!
        def my_function():
            log.info("This is an informational message.")
            log.debug("This is a debug message.")

        # You can change the log level at runtime if needed.
        log.set_level("DEBUG")
        my_function()
    """

    _instance = None
    _initialized: bool

    def __new__(cls, *args: Any, **kwargs: Any) -> "Logger":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        log_dir: str = "logs",
        level: str = "INFO",
        log_console: bool = False,
        use_timestamp: bool = True,
        wandb_config: Optional[Dict[str, Any]] = None,
        tensorboard_config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize and configure the logger. This method runs only once."""
        if hasattr(self, "_initialized") and self._initialized:
            return

        self._logger = logging.getLogger("CuMindLogger")
        self.tb_writer: Optional[Any] = None

        # Single formatter for all handlers
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%I:%M:%S %p")  # %(name)s sub logger

        # Setup file handler
        if use_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_dir = Path(log_dir) / timestamp
        else:
            self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(self.log_dir / "training.log")
        file_handler.setFormatter(formatter)
        self._logger.addHandler(file_handler)

        # Setup console handler if requested
        if log_console:
            stdout_handler = logging.StreamHandler(sys.stdout)
            stdout_handler.setFormatter(formatter)
            self._logger.addHandler(stdout_handler)

        self.set_level(level)

        # Initialize integrations
        self.use_wandb = wandb_config is not None
        self.use_tensorboard = tensorboard_config is not None
        if self.use_wandb:
            assert wandb_config is not None
            if wandb.run is None:
                wandb.init(**wandb_config)
        if self.use_tensorboard:
            self.tb_writer = tb.summary.create_file_writer(str(self.log_dir))

        self._initialized = True
        self.info(f"Unified logger initialized. Logging to: {self.log_dir}")

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Logs a message with level DEBUG."""
        self._logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Logs a message with level INFO."""
        self._logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Logs a message with level WARNING."""
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Logs a message with level ERROR."""
        self._logger.error(msg, *args, **kwargs)

    def exception(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Logs a message with level ERROR, including exception info. Place in except block."""
        self._logger.exception(msg, *args, **kwargs)

    def critical(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Logs a message with level CRITICAL."""
        self._logger.critical(msg, *args, **kwargs)

    def log(self, level: int, msg: str, *args: Any, **kwargs: Any) -> None:
        """Logs a message with the specified level."""
        self._logger.log(level, msg, *args, **kwargs)

    def log_scalar(self, name: str, value: float, step: int) -> None:
        """Log a scalar value to the console and to W&B."""
        self.info(f"Step {step:4d}: {name} = {value:.6f}")
        if self.use_wandb:
            wandb.log({name: value}, step=step)
        if self.use_tensorboard and self.tb_writer:
            with self.tb_writer.as_default():
                tb.summary.scalar(name, value, step=step)

    def log_scalars(self, metrics: Dict[str, float], step: int) -> None:
        """Log multiple scalar values."""
        for name, value in metrics.items():
            self.log_scalar(name, value, step)

    def set_level(self, level: str) -> None:
        """Change the logging level at runtime.
        Args:
            level: The new logging level (e.g., "DEBUG", "INFO").
        """
        self.info(f"Changing logger level to {level.upper()}")
        self._logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    def close(self) -> None:
        """Close logger and cleanup resources, like file handlers and W&B run."""
        if self.use_wandb and wandb.run is not None:
            wandb.finish()
        if self.use_tensorboard and self.tb_writer is not None:
            self.tb_writer.close()

        self.info("Closing logger handlers and shutting down logging system.")
        for handler in self._logger.handlers[:]:
            handler.close()
            self._logger.removeHandler(handler)

        logging.shutdown()
