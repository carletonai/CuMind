"""Replay buffer implementation with optional prioritization."""

import random
from collections import deque
from typing import Any, List, Tuple

import numpy as np


class ReplayBuffer:
    """Replay buffer for storing and sampling game trajectories."""

    def __init__(self, capacity: int, prioritized: bool = False):
        """Initialize replay buffer.

        Args:
            capacity: Maximum number of trajectories to store
            prioritized: Whether to use prioritized experience replay

        Implementation:
            - Create deque with maxlen=capacity for storage
            - Setup prioritization if enabled (sum tree or similar)
            - Initialize internal counters and indices
        """
        # Branch: feature/replay-buffer-init
        raise NotImplementedError("ReplayBuffer.__init__ needs to be implemented")

    def add(self, trajectory: Any) -> None:
        """Add a trajectory to the buffer.

        Args:
            trajectory: Game trajectory data

        Implementation:
            - Append trajectory to internal storage
            - Update priority if using prioritized replay
            - Handle capacity overflow (deque auto-removes oldest)
        """
        # Branch: feature/replay-buffer-add
        raise NotImplementedError("ReplayBuffer.add needs to be implemented")

    def sample(self, batch_size: int) -> List[Any]:
        """Sample a batch of trajectories.

        Args:
            batch_size: Number of trajectories to sample

        Returns:
            List of sampled trajectories

        Implementation:
            - Use random.sample() for uniform sampling
            - Use priority-based sampling if prioritized=True
            - Return list of trajectory objects
        """
        # Branch: feature/replay-buffer-sample
        raise NotImplementedError("ReplayBuffer.sample needs to be implemented")

    def __len__(self) -> int:
        """Get number of trajectories in buffer.

        Returns:
            Current number of stored trajectories

        Implementation:
            - Return length of internal storage container
        """
        # Branch: feature/replay-buffer-len
        raise NotImplementedError("ReplayBuffer.__len__ needs to be implemented")

    def is_ready(self, min_size: int) -> bool:
        """Check if buffer has enough samples for training.

        Args:
            min_size: Minimum number of samples required

        Returns:
            True if buffer has at least min_size samples

        Implementation:
            - Compare current buffer length with min_size
            - Return boolean result
        """
        # Branch: feature/replay-buffer-ready
        raise NotImplementedError("ReplayBuffer.is_ready needs to be implemented")
