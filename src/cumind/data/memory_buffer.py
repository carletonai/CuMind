"""Memory buffer implementations with unified API for storing and sampling data samples."""

import random
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Deque, List, Optional, Union

import numpy as np


class MemoryBuffer(ABC):
    """Abstract base class for memory buffers with unified API."""

    def __init__(self, capacity: int):
        """Initialize memory buffer.

        Args:
            capacity: Maximum number of samples to store
        """
        self.capacity = capacity
        self.buffer: Deque[Any] = deque(maxlen=capacity)

    @abstractmethod
    def add(self, sample: Any) -> None:
        """Add a sample to the buffer.

        Args:
            sample: Data sample to store
        """
        pass

    @abstractmethod
    def sample(self, batch_size: int) -> List[Any]:
        """Sample a batch of data.

        Args:
            batch_size: Number of samples to retrieve

        Returns:
            List of sampled data
        """
        pass

    def __len__(self) -> int:
        """Get number of samples in buffer.

        Returns:
            Current number of stored samples
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

    def clear(self) -> None:
        """Clear all samples from the buffer."""
        self.buffer.clear()


class ReplayBuffer(MemoryBuffer):
    """Standard replay buffer with uniform sampling."""

    def __init__(self, capacity: int):
        """Initialize replay buffer.

        Args:
            capacity: Maximum number of samples to store
        """
        super().__init__(capacity)

    def add(self, sample: Any) -> None:
        """Add a sample to the buffer.

        Args:
            sample: Data sample to store
        """
        self.buffer.append(sample)

    def sample(self, batch_size: int) -> List[Any]:
        """Sample a batch of data using uniform sampling.

        Args:
            batch_size: Number of samples to retrieve

        Returns:
            List of sampled data
        """
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        return random.sample(list(self.buffer), batch_size)


class PrioritizedReplayBuffer(MemoryBuffer):
    """Prioritized replay buffer with importance sampling."""

    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4, epsilon: float = 1e-6):
        """Initialize prioritized replay buffer.

        Args:
            capacity: Maximum number of samples to store
            alpha: Priority exponent (0 = uniform, 1 = greedy)
            beta: Importance sampling exponent for bias correction
            epsilon: Small value to avoid zero priorities
        """
        super().__init__(capacity)
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.priorities: Deque[float] = deque(maxlen=capacity)
        self.max_priority = 1.0

    def add(self, sample: Any) -> None:
        """Add a sample to the buffer with maximum priority.

        Args:
            sample: Data sample to store
        """
        self.buffer.append(sample)
        self.priorities.append(self.max_priority)

    def sample(self, batch_size: int) -> List[Any]:
        """Sample a batch of data using priority-based sampling.

        Args:
            batch_size: Number of samples to retrieve

        Returns:
            List of sampled data
        """
        if len(self.buffer) < batch_size:
            return list(self.buffer)

        # Calculate sampling probabilities
        priorities = np.array(self.priorities)
        priorities_alpha = np.power(priorities + self.epsilon, self.alpha)
        probabilities = priorities_alpha / np.sum(priorities_alpha)

        # Sample indices
        indices = np.random.choice(len(self.buffer), size=batch_size, p=probabilities, replace=False)
        return [self.buffer[i] for i in indices]

    def update_priorities(self, indices: List[int], priorities: List[float]) -> None:
        """Update priorities for sampled data.

        Args:
            indices: Indices of samples to update
            priorities: New priority values
        """
        for idx, priority in zip(indices, priorities):
            if idx < len(self.priorities):
                self.priorities[idx] = priority
                self.max_priority = max(self.max_priority, priority)

    def clear(self) -> None:
        """Clear all samples and priorities from the buffer."""
        super().clear()
        self.priorities.clear()
        self.max_priority = 1.0


class TreeBuffer(MemoryBuffer):
    """Tree-based buffer for efficient priority sampling using sum tree."""

    def __init__(self, capacity: int, alpha: float = 0.6):
        """Initialize tree buffer.

        Args:
            capacity: Maximum number of samples to store
            alpha: Priority exponent (0 = uniform, 1 = greedy)
        """
        super().__init__(capacity)
        self.alpha = alpha
        self.epsilon = 1e-6
        self.max_priority = 1.0

        # Initialize sum tree
        self.tree_size = 1
        while self.tree_size < capacity:
            self.tree_size *= 2

        self.sum_tree = np.zeros(2 * self.tree_size - 1)
        self.data_pointer = 0
        self.n_entries = 0

    def _propagate(self, idx: int, change: float) -> None:
        """Propagate priority change up the tree.

        Args:
            idx: Tree index to update
            change: Priority change value
        """
        parent = (idx - 1) // 2
        self.sum_tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        """Retrieve index for given priority sum.

        Args:
            idx: Current tree index
            s: Priority sum to search for

        Returns:
            Index of the leaf node
        """
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.sum_tree):
            return idx

        if s <= self.sum_tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.sum_tree[left])

    def add(self, sample: Any) -> None:
        """Add a sample to the buffer with maximum priority.

        Args:
            sample: Data sample to store
        """
        self.buffer.append(sample)

        # Update sum tree
        tree_idx = self.data_pointer + self.tree_size - 1
        self._update(tree_idx, self.max_priority)

        self.data_pointer = (self.data_pointer + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def _update(self, tree_idx: int, priority: float) -> None:
        """Update priority in the sum tree.

        Args:
            tree_idx: Tree index to update
            priority: New priority value
        """
        change = priority - self.sum_tree[tree_idx]
        self.sum_tree[tree_idx] = priority
        self._propagate(tree_idx, change)

    def sample(self, batch_size: int) -> List[Any]:
        """Sample a batch of data using sum tree.

        Args:
            batch_size: Number of samples to retrieve

        Returns:
            List of sampled data
        """
        if len(self.buffer) < batch_size:
            return list(self.buffer)

        sampled_data = []
        segment = self.sum_tree[0] / batch_size

        for i in range(batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            idx = self._retrieve(0, s)
            data_idx = idx - self.tree_size + 1
            sampled_data.append(self.buffer[data_idx])

        return sampled_data

    def update_priorities(self, indices: List[int], priorities: List[float]) -> None:
        """Update priorities for sampled data.

        Args:
            indices: Indices of samples to update
            priorities: New priority values
        """
        for idx, priority in zip(indices, priorities):
            if idx < self.n_entries:
                tree_idx = idx + self.tree_size - 1
                priority_alpha = np.power(priority + self.epsilon, self.alpha)
                self._update(tree_idx, priority_alpha)
                self.max_priority = max(self.max_priority, priority_alpha)

    def clear(self) -> None:
        """Clear all samples and reset sum tree."""
        super().clear()
        self.sum_tree.fill(0)
        self.data_pointer = 0
        self.n_entries = 0
        self.max_priority = 1.0
