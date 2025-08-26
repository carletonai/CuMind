"""Unified logger with TensorBoard and Weights & Biases support."""

import logging
import sys
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

from cumind.utils.config import cfg


class Logger:
    """A singleton logger that provides a unified, configurable interface."""

    _instance: Optional["Logger"] = None
    _initialized: bool = False
    _lock: threading.RLock = threading.RLock()

    @classmethod
    def _get_instance(cls) -> "Logger":
        """Get the singleton instance, creating it if necessary."""
        if cls._instance is None:
            cls()
        assert cls._instance is not None
        return cls._instance

    def __new__(cls, *args: Any, **kwargs: Any) -> "Logger":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Constructor - initialization is handled in __new__."""
        pass

    def boot(self, cfg: cfg, workspace: Path) -> None:
        """Initialize the logger instance."""
        if type(self)._initialized:
            return

        level: str = cfg.logging.level
        console: bool = cfg.logging.console
        timestamps: bool = cfg.logging.timestamps

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

        # Setup directories
        self.start_time = datetime.now()
        self.workspace_path = workspace

        # Setup file handler
        file_handler = logging.FileHandler(self.workspace_path / "training.log")
        file_handler.setFormatter(self._formatter)
        self._logger.addHandler(file_handler)

        # Setup console handler if requested
        if console:
            self.open()

        self.set_level(level)

        # Setup wandb config
        self.use_wandb = cfg.logging.wandb
        if self.use_wandb:
            import os

            os.environ["WANDB_DIR"] = str(self.workspace_path)
            import wandb

            wandb_config: Dict[str, Any] = {
                "project": "CuMind",
                "name": cfg.logging.title,
                "tags": cfg.logging.tags,
                "monitor_gym": True,  # hard coded?
            }
            if wandb.run is None:
                wandb.init(**wandb_config)
        type(self)._initialized = True

    @classmethod
    def get_timestamp(cls) -> datetime:
        """Returns the timestamp indicating when the logger was started."""
        return cls._get_instance().start_time

    @classmethod
    def get_workspace(cls) -> Path:
        """Returns the workspace path used by the logger."""
        return cls._get_instance().workspace_path

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
    def log_scalar(cls, name: str, value: float, episode: int) -> None:
        instance = cls._get_instance()
        cls.debug(f"Episode {episode:4d}: {name} = {value:.6f}")
        if instance.use_wandb:
            import wandb

            wandb.log({f"episode/{name}": value}, step=episode)

    @classmethod
    def log_scalars(cls, metrics: Dict[str, float], episode: int) -> None:
        for name, value in metrics.items():
            cls.log_scalar(name, value, episode)

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
        """Open console output for logging to stdout, safely handling missing stdout."""
        with cls._lock:
            instance = cls._get_instance()
            if instance._console_handler is None:
                stream = sys.stdout if sys.stdout is not None else getattr(sys, "__stdout__", None)
                if stream is None:
                    instance._logger.warning("No stdout available; console handler not added.")
                    return
                instance._console_handler = logging.StreamHandler(stream)
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
        start = cls.get_timestamp()
        end = datetime.now()
        return end - start

    @classmethod
    def shutdown(cls) -> None:
        """Close logger and cleanup resources."""
        instance = cls._get_instance()

        elapsed = instance.elapsed()
        instance._logger.info(f"Logging Session ran for {elapsed}.")

        if instance.use_wandb:
            import wandb

            if wandb.run is not None:
                wandb.finish()

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


class TqdmSink:
    def __init__(self, mode: bool):
        self.mode = mode
        if mode:
            self.sink = self._stdout_sink
        else:
            self.sink = self._logger_sink

    def write(self, msg: Any) -> None:
        self.sink(msg)

    def _stdout_sink(self, msg: Any) -> None:
        sys.stdout.write(str(msg))

    def _logger_sink(self, msg: Any) -> None:
        log.info(str(msg))
