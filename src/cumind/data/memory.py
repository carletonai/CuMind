"""Memory buffer implementations for storing and sampling training data."""

import random
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Deque, Dict, List

import numpy as np

from cumind.utils.logger import log


class Memory(ABC):
    """Abstract base class for memory buffers."""

    def __init__(self, capacity: int):
        """Initializes the memory buffer.

        Args:
            capacity: Maximum number of samples to store.
        """
        log.info(f"Initializing memory buffer with capacity {capacity}.")
        self.capacity = capacity
        self.buffer: Deque[Any] = deque(maxlen=capacity)

    @abstractmethod
    def add(self, sample: List[Dict[str, Any]]) -> None:
        """Adds a sample to the buffer.

        Args:
            sample: The data sample to store.
        """
        raise NotImplementedError

    @abstractmethod
    def sample(self, batch_size: int) -> List[Any]:
        """Samples a batch of data from the buffer.

        Args:
            batch_size: The number of samples to retrieve.

        Returns:
            A list of sampled data points.
        """
        raise NotImplementedError

    def __len__(self) -> int:
        """Returns the current number of samples in the buffer."""
        return len(self.buffer)

    def get_size(self) -> int:
        """Returns the current size of the buffer."""
        return len(self)

    def get_pct(self) -> float:
        """Returns the percentage of the buffer that is filled."""
        if self.capacity == 0:
            log.critical("Buffer capacity is zero, cannot calculate percentage.")
            raise ValueError("Buffer capacity is zero, cannot calculate percentage.")
        return float((len(self) / self.capacity) * 100)

    def is_ready(self, min_size: int = 0, min_fill_pct: float = 0) -> bool:
        """Checks if the buffer has enough samples for training.

        Args:
            min_size: The minimum number of samples required.

        Returns:
            True if the buffer contains at least `min_size` samples.
        """
        if min_size < 0 or min_fill_pct < 0 or min_fill_pct > 1:
            log.critical(f"Invalid arguments for is_ready: min_size={min_size}, min_fill_pct={min_fill_pct}")
            raise ValueError("min_size must be non-negative and min_fill_pct must be between 0 and 1.")
        if self.capacity == 0:
            log.critical("Buffer capacity is zero, cannot check readiness.")
            raise ValueError("Buffer capacity is zero, cannot check readiness.")
        if min_size == 0 and min_fill_pct == 0:
            log.critical("At least one of min_size or min_fill_pct must be greater than zero.")
            raise ValueError("At least one of min_size or min_fill_pct must be greater than zero.")
        return (len(self) >= min_size) and (len(self) >= min_fill_pct * self.capacity)

    def clear(self) -> None:
        """Removes all samples from the buffer."""
        log.info(f"Clearing buffer of size {len(self)}.")
        self.buffer.clear()


class MemoryBuffer(Memory):
    """A standard memory buffer with uniform sampling."""

    def __init__(self, capacity: int):
        """Initializes the memory buffer with capacity."""
        super().__init__(capacity)

    def add(self, sample: List[Dict[str, Any]]) -> None:
        """Adds a sample to the buffer.

        Args:
            sample: The data sample to store.
        """
        log.debug(f"Adding sample to MemoryBuffer. Current size: {len(self)}.")
        self.buffer.append(sample)

    def sample(self, batch_size: int) -> List[Any]:
        """Samples a batch of data using uniform random sampling.

        Args:
            batch_size: The number of samples to retrieve.

        Returns:
            A list of sampled data points.
        """
        log.debug(f"Sampling {batch_size} samples from MemoryBuffer.")
        if len(self.buffer) < batch_size:
            log.warning(f"Requested batch size {batch_size} is larger than buffer size {len(self)}. Returning all samples.")
            return list(self.buffer)
        return random.sample(list(self.buffer), batch_size)


class PrioritizedMemoryBuffer(Memory):
    """A memory buffer that uses priority-based sampling."""

    def __init__(self, capacity: int, alpha: float, epsilon: float, beta: float):
        """Initializes the prioritized memory buffer.

        Args:
            capacity: Maximum number of samples to store.
            alpha: The priority exponent (0=uniform, 1=greedy).
            beta: The importance sampling exponent for bias correction.
            epsilon: A small value to avoid zero priorities.
        """
        super().__init__(capacity)
        log.info(f"Initializing PrioritizedMemoryBuffer with capacity {capacity}, alpha={alpha}, epsilon={epsilon}, beta={beta}.")
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.priorities: Deque[float] = deque(maxlen=capacity)
        self.max_priority = 1.0

    def add(self, sample: List[Dict[str, Any]]) -> None:
        """Adds a sample to the buffer with maximum priority.

        Args:
            sample: The data sample to store.
        """
        log.debug(f"Adding sample to PrioritizedMemoryBuffer with max priority. Current size: {len(self)}.")
        self.buffer.append(sample)
        self.priorities.append(self.max_priority)

    def sample(self, batch_size: int) -> List[Any]:
        """Samples a batch of data using priority-based sampling.

        Args:
            batch_size: The number of samples to retrieve.

        Returns:
            A list of sampled data points.
        """
        log.debug(f"Sampling {batch_size} samples from PrioritizedMemoryBuffer.")
        if len(self.buffer) < batch_size:
            log.warning(f"Requested batch size {batch_size} is larger than buffer size {len(self)}. Returning all samples.")
            return list(self.buffer)

        priorities = np.array(self.priorities)
        probabilities = (priorities**self.alpha) / np.sum(priorities**self.alpha)

        indices = np.random.choice(len(self.buffer), size=batch_size, p=probabilities, replace=False)
        return [self.buffer[i] for i in indices]

    def update_priorities(self, indices: List[int], priorities: List[float]) -> None:
        """Updates the priorities of sampled data.

        Args:
            indices: The indices of the samples to update.
            priorities: The new priority values.
        """
        log.debug(f"Updating priorities for {len(indices)} samples in PrioritizedMemoryBuffer.")
        for idx, priority in zip(indices, priorities):
            if idx < len(self.priorities):
                self.priorities[idx] = priority
                self.max_priority = max(self.max_priority, priority)

    def clear(self) -> None:
        """Removes all samples and priorities from the buffer."""
        log.info(f"Clearing PrioritizedMemoryBuffer of size {len(self)}.")
        super().clear()
        self.priorities.clear()
        self.max_priority = 1.0


class TreeBuffer(Memory):
    """A tree-based buffer for efficient priority sampling using a sum tree."""

    def __init__(self, capacity: int, alpha: float, epsilon: float):
        """Initializes the tree buffer.

        Args:
            capacity: Maximum number of samples to store.
            alpha: The priority exponent (0=uniform, 1=greedy).
            epsilon: A small value to avoid zero priorities.
        """
        super().__init__(capacity)
        log.info(f"Initializing TreeBuffer with capacity {capacity}, alpha={alpha}, epsilon={epsilon}.")
        self.alpha = alpha
        self.epsilon = epsilon
        self.max_priority = 1.0

        self.tree_size = 1
        while self.tree_size < capacity:
            self.tree_size *= 2

        self.sum_tree = np.zeros(2 * self.tree_size - 1)
        self.data_pointer = 0
        self.n_entries = 0

    def _propagate(self, idx: int, change: float) -> None:
        """Propagates a priority change up the sum tree."""
        parent = (idx - 1) // 2
        self.sum_tree[parent] += change
        if idx > 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        """Retrieves the leaf index for a given priority sum."""
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.sum_tree):
            return idx

        if s <= self.sum_tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.sum_tree[left])

    def add(self, sample: List[Dict[str, Any]]) -> None:
        """Adds a sample to the buffer with maximum priority.

        Args:
            sample: The data sample to store.
        """
        log.debug(f"Adding sample to TreeBuffer. Current size: {self.n_entries}.")
        self.buffer.append(sample)

        tree_idx = self.data_pointer + self.tree_size - 1
        self._update(tree_idx, self.max_priority)

        self.data_pointer = (self.data_pointer + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def _update(self, tree_idx: int, priority: float) -> None:
        """Updates a priority value in the sum tree."""
        change = priority - self.sum_tree[tree_idx]
        self.sum_tree[tree_idx] = priority
        self._propagate(tree_idx, change)

    def sample(self, batch_size: int) -> List[Any]:
        """Samples a batch of data using the sum tree.

        Args:
            batch_size: The number of samples to retrieve.

        Returns:
            A list of sampled data points.
        """
        if len(self.buffer) < batch_size:
            log.warning(f"Requested batch size {batch_size} is larger than buffer size {len(self)}. Returning all samples.")
            return list(self.buffer)

        log.debug(f"Sampling {batch_size} samples from TreeBuffer.")
        sampled_data = []
        segment = self.sum_tree[0] / batch_size

        for i in range(batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            idx = self._retrieve(0, s)
            data_idx = idx - self.tree_size + 1
            sampled_data.append(self.buffer[data_idx])

        return sampled_data

    def update_priorities(self, indices: List[int], priorities: List[float]) -> None:
        """Updates the priorities of sampled data in the sum tree.

        Args:
            indices: The indices of the samples to update.
            priorities: The new priority values.
        """
        log.debug(f"Updating priorities for {len(indices)} samples in TreeBuffer.")
        for tree_idx, priority in zip(indices, priorities):
            p = (priority + self.epsilon) ** self.alpha
            self._update(tree_idx, p)
            self.max_priority = max(self.max_priority, p)

    def clear(self) -> None:
        """Removes all samples and resets the sum tree."""
        log.info(f"Clearing TreeBuffer of size {self.n_entries}.")
        super().clear()
        self.sum_tree.fill(0)
        self.data_pointer = 0
        self.n_entries = 0
        self.max_priority = 1.0
