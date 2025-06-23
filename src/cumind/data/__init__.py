"""Data module for replay buffers and self-play."""

from .replay_buffer import ReplayBuffer
from .self_play import SelfPlay

__version__ = "0.0.1"
__all__ = ["ReplayBuffer", "SelfPlay"]
