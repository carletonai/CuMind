"""Comprehensive tests for data components (MemoryBuffer, SelfPlay)."""

import gymnasium as gym
import pytest

from cumind.agent.agent import Agent
from cumind.config import Config
from cumind.data.memory import MemoryBuffer, PrioritizedMemoryBuffer, TreeBuffer
from cumind.data.self_play import SelfPlay


def _create_buffer(BufferClass, capacity, config):  # noqa: N803
    """Helper function to create a buffer with the correct arguments."""
    if BufferClass == PrioritizedMemoryBuffer:
        return BufferClass(capacity=capacity, alpha=config.memory_per_alpha, epsilon=config.memory_per_epsilon, beta=config.memory_per_beta)
    if BufferClass == TreeBuffer:
        return BufferClass(capacity=capacity, alpha=config.memory_per_alpha, epsilon=config.memory_per_epsilon)
    return BufferClass(capacity=capacity)


class TestMemoryBuffer:
    """Test suite for memory buffers."""

    @pytest.mark.parametrize("BufferClass", [MemoryBuffer, PrioritizedMemoryBuffer, TreeBuffer])
    def test_buffer_initialization(self, BufferClass):  # noqa: N803
        """Test memory buffer initialization."""
        capacity = 100
        config = Config()
        buffer = _create_buffer(BufferClass, capacity, config)
        assert buffer.capacity == capacity
        assert len(buffer) == 0

    @pytest.mark.parametrize("BufferClass", [MemoryBuffer, PrioritizedMemoryBuffer, TreeBuffer])
    def test_add_and_len(self, BufferClass):  # noqa: N803
        """Test adding samples and checking buffer length."""
        config = Config()
        buffer = _create_buffer(BufferClass, 10, config)
        buffer.add({"obs": 1, "action": 0})
        assert len(buffer) == 1
        buffer.add({"obs": 2, "action": 1})
        assert len(buffer) == 2

    @pytest.mark.parametrize("BufferClass", [MemoryBuffer, PrioritizedMemoryBuffer, TreeBuffer])
    def test_buffer_capacity(self, BufferClass):  # noqa: N803
        """Test that buffer does not exceed capacity."""
        capacity = 5
        config = Config()
        buffer = _create_buffer(BufferClass, capacity, config)
        for i in range(10):
            buffer.add({"obs": i})
        assert len(buffer) == capacity

    @pytest.mark.parametrize("BufferClass", [MemoryBuffer, PrioritizedMemoryBuffer, TreeBuffer])
    def test_is_ready(self, BufferClass):  # noqa: N803
        """Test buffer readiness check."""
        config = Config()
        buffer = _create_buffer(BufferClass, 20, config)
        assert not buffer.is_ready(min_size=10)
        for i in range(10):
            buffer.add({"obs": i})
        assert buffer.is_ready(min_size=10)
        assert not buffer.is_ready(min_size=11)

    @pytest.mark.parametrize("BufferClass", [MemoryBuffer, PrioritizedMemoryBuffer, TreeBuffer])
    def test_clear_buffer(self, BufferClass):  # noqa: N803
        """Test clearing the buffer."""
        config = Config()
        buffer = _create_buffer(BufferClass, 10, config)
        for i in range(5):
            buffer.add({"obs": i})
        buffer.clear()
        assert len(buffer) == 0

    @pytest.mark.parametrize("BufferClass", [MemoryBuffer, PrioritizedMemoryBuffer, TreeBuffer])
    def test_sample_from_buffer(self, BufferClass):  # noqa: N803
        """Test sampling from the buffer."""
        config = Config()
        buffer = _create_buffer(BufferClass, 20, config)
        for i in range(15):
            buffer.add({"id": i})

        result = buffer.sample(batch_size=5)
        if isinstance(result, tuple):
            sample, _ = result
        else:
            sample = result
        assert isinstance(sample, list)
        assert len(sample) == 5

        # Check for unique samples if possible
        if isinstance(buffer, MemoryBuffer):
            ids = {item["id"] for item in sample}
            assert len(ids) == 5


class TestSelfPlay:
    """Test suite for the SelfPlay class."""

    def test_self_play_initialization(self):
        """Test SelfPlay initialization."""
        config = Config()
        agent = Agent(config)
        memory_buffer = MemoryBuffer(capacity=100)
        self_play = SelfPlay(config, agent, memory_buffer)

        assert self_play.config == config
        assert self_play.agent == agent
        assert self_play.memory == memory_buffer

    def test_run_episode(self):
        """Test running a single self-play episode."""
        config = Config()
        config.env_action_space_size = 2
        config.env_observation_shape = (4,)
        agent = Agent(config)
        memory_buffer = MemoryBuffer(capacity=100)
        env = gym.make("CartPole-v1")
        self_play = SelfPlay(config, agent, memory_buffer)

        episode_data = self_play.run_episode(env)

        assert isinstance(episode_data, tuple)
        assert len(episode_data[2]) > 0
        assert len(memory_buffer) == 1  # One episode was added

        # Check experience format
        for step in episode_data[2]:
            assert "observation" in step
            assert "action" in step
            assert "reward" in step
            assert "policy" in step
            assert "done" in step

    def test_collect_samples(self):
        """Test collecting samples from multiple episodes."""
        config = Config()
        config.env_action_space_size = 2
        config.env_observation_shape = (4,)
        agent = Agent(config)
        memory_buffer = MemoryBuffer(capacity=100)
        env = gym.make("CartPole-v1")
        self_play = SelfPlay(config, agent, memory_buffer)

        num_episodes = 3
        self_play.collect_samples(env, num_episodes)
        assert len(memory_buffer) == num_episodes


if __name__ == "__main__":
    pytest.main([__file__])
