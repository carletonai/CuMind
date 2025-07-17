"""CuMind: A modular reinforcement learning framework."""

from .agent import Agent, Trainer
from .runner import inference, train
from .utils import log
from .utils.config import cfg
from .utils.prng import key

__version__ = "0.1.7"
__all__ = ["Agent", "Trainer", "train", "inference", "cfg", "log", "key"]
