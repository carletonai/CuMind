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
        """
        self.capacity = capacity
        self.prioritized = prioritized
        self.buffer = deque(maxlen=capacity)

        if prioritized:
            # For simplicity, we'll implement a basic prioritized buffer
            # In a full implementation, you'd use a sum tree for efficiency
            self.priorities = deque(maxlen=capacity)
        else:
            self.priorities = None

    def add(self, trajectory: Any) -> None:
        """Add a trajectory to the buffer.

        Args:
            trajectory: Game trajectory data
        """
        self.buffer.append(trajectory)

        if self.prioritized:
            # Assign max priority to new trajectories
            if self.priorities:
                max_priority = max(self.priorities)
            else:
                max_priority = 1.0
            self.priorities.append(max_priority)

    def sample(self, batch_size: int) -> List[Any]:
        """Sample a batch of trajectories.

        Args:
            batch_size: Number of trajectories to sample

        Returns:
            List of sampled trajectories
        """
        if len(self.buffer) < batch_size:
            return list(self.buffer)

        if self.prioritized and self.priorities:
            # Priority-based sampling
            priorities = np.array(self.priorities)
            probabilities = priorities / np.sum(priorities)
            indices = np.random.choice(len(self.buffer), size=batch_size, p=probabilities)
            return [self.buffer[i] for i in indices]
        else:
            # Uniform sampling
            return random.sample(list(self.buffer), batch_size)

    def __len__(self) -> int:
        """Get number of trajectories in buffer.

        Returns:
            Current number of stored trajectories
        """
        return len(self.buffer)

    def is_ready(self, min_size: int) -> bool:
        """Check if buffer has enough samples for training.

        Args:
            min_size: Minimum number of samples required

        Returns:
            True if buffer has at least min_size samples
        """
        return len(self.buffer) >= min_size
