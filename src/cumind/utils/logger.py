"""Unified logger with TensorBoard and Weights & Biases support."""

import logging
import sys
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import tensorboard as tb  # type: ignore
import wandb

from cumind.utils.config import cfg


class Logger:
    """A singleton logger that provides a unified, configurable interface."""

    _instance: Optional["Logger"] = None
    _initialized: bool = False
    _lock: threading.RLock = threading.RLock()

    def __new__(cls, *args: Any, **kwargs: Any) -> "Logger":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
                    # Initialize the instance immediately
                    cls._instance._initialize(*args, **kwargs)
        return cls._instance

    def _initialize(
        self,
        cfg: Optional[cfg] = None,
        dir: str = "logs",
        level: str = "INFO",
        console: bool = False,
        timestamps: bool = True,
        wandb_config: Optional[Dict[str, Any]] = None,
        tensorboard_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the logger instance."""

        if cfg is not None:
            dir = cfg.logging.dir
            level = cfg.logging.level
            console = cfg.logging.console
            timestamps = cfg.logging.timestamps

        self._logger = logging.getLogger("CuMindLogger")
        self.tb_writer: Optional[Any] = None
        self._console_handler: Optional[logging.StreamHandler[Any]] = None

        if not timestamps:
            self.FORMAT = "%(levelname)s - %(message)s"
            self.DATEFMT = ""
        else:
            self.FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
            self.DATEFMT = "%I:%M:%S %p"

        # Single formatter for all handlers
        self._formatter = logging.Formatter(self.FORMAT, datefmt=self.DATEFMT)

        # Setup file handler

        self.start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = Path(dir) / self.start_time
        self.log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(self.log_dir / "training.log")
        file_handler.setFormatter(self._formatter)
        self._logger.addHandler(file_handler)

        # Setup console handler if requested
        if console:
            self.open()

        self.set_level(level)

        # Integrations
        self.use_wandb = wandb_config is not None
        self.use_tensorboard = tensorboard_config is not None
        if self.use_wandb:
            assert wandb_config is not None
            if wandb.run is None:
                wandb.init(**wandb_config)
        if self.use_tensorboard:
            self.tb_writer = tb.summary.create_file_writer(str(self.log_dir))

        type(self)._initialized = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Constructor - initialization is handled in __new__."""
        pass

    @classmethod
    def _get_instance(cls) -> "Logger":
        """Get the singleton instance, creating it if necessary."""
        if cls._instance is None:
            cls()
        assert cls._instance is not None
        return cls._instance

    @classmethod
    def debug(cls, msg: str, *args: Any, **kwargs: Any) -> None:
        cls._get_instance()._logger.debug(msg, *args, **kwargs)

    @classmethod
    def info(cls, msg: str, *args: Any, **kwargs: Any) -> None:
        cls._get_instance()._logger.info(msg, *args, **kwargs)

    @classmethod
    def warning(cls, msg: str, *args: Any, **kwargs: Any) -> None:
        cls._get_instance()._logger.warning(msg, *args, **kwargs)

    @classmethod
    def error(cls, msg: str, *args: Any, **kwargs: Any) -> None:
        cls._get_instance()._logger.error(msg, *args, **kwargs)

    @classmethod
    def exception(cls, msg: str, *args: Any, **kwargs: Any) -> None:
        cls._get_instance()._logger.exception(msg, *args, **kwargs)

    @classmethod
    def critical(cls, msg: str, *args: Any, **kwargs: Any) -> None:
        cls._get_instance()._logger.critical(msg, *args, **kwargs)

    @classmethod
    def log_scalar(cls, name: str, value: float, step: int) -> None:
        instance = cls._get_instance()
        cls.info(f"Step {step:4d}: {name} = {value:.6f}")
        if instance.use_wandb:
            wandb.log({name: value}, step=step)
        if instance.use_tensorboard and instance.tb_writer:
            with instance.tb_writer.as_default():
                tb.summary.scalar(name, value, step=step)

    @classmethod
    def log_scalars(cls, metrics: Dict[str, float], step: int) -> None:
        for name, value in metrics.items():
            cls.log_scalar(name, value, step)

    @classmethod
    def set_level(cls, level: str) -> None:
        """Change the logging level at runtime.
        Args:
            level: The new logging level (e.g., "DEBUG", "INFO").
        """
        instance = cls._get_instance()
        instance._logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        cls.info(f"Logger level set to {level.upper()}")

    @classmethod
    def open(cls) -> None:
        """Open console output for logging to stdout."""
        with cls._lock:
            instance = cls._get_instance()
            if instance._console_handler is None:
                instance._console_handler = logging.StreamHandler(sys.stdout)
                # Use instance's FORMAT/DATEFMT, not class variables
                instance._console_handler.setFormatter(ColorFormatter(instance.FORMAT, datefmt=instance.DATEFMT))
                instance._logger.addHandler(instance._console_handler)
                cls.info("Console output opened")

    @classmethod
    def close(cls) -> None:
        """Close console output for logging to stdout."""
        with cls._lock:
            instance = cls._get_instance()
            if instance._console_handler is not None:
                instance._logger.removeHandler(instance._console_handler)
                instance._console_handler.close()
                instance._console_handler = None
                cls.info("Console output closed")

    @classmethod
    def elapsed(cls) -> timedelta:
        """Return the elapsed time since logger start as a timedelta."""
        start = datetime.strptime(cls._get_instance().start_time, "%Y%m%d_%H%M%S")
        end = datetime.now()
        return end - start

    @classmethod
    def shutdown(cls) -> None:
        """Close logger and cleanup resources."""
        instance = cls._get_instance()

        elapsed = instance.elapsed()
        instance._logger.info(f"Logging Session ran for {elapsed}.")

        if instance.use_wandb and wandb.run is not None:
            wandb.finish()
        if instance.use_tensorboard and instance.tb_writer is not None:
            instance.tb_writer.close()

        cls.info("Closing logger handlers and shutting down logging system.")
        for handler in instance._logger.handlers[:]:
            handler.close()
            instance._logger.removeHandler(handler)
        logging.shutdown()


class ColorFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[41m",  # Red background
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        msg = super().format(record)
        return f"{color}{msg}{self.RESET}"


# Alias
log = Logger
