"""Comprehensive tests for data components (MemoryBuffer, SelfPlay)."""

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cumind.agent.agent import Agent
from cumind.config import Config
from cumind.data.memory_buffer import MemoryBuffer, PrioritizedReplayBuffer, ReplayBuffer, TreeBuffer
from cumind.data.self_play import SelfPlay


class TestMemoryBuffer:
    """Test suite for memory buffers."""

    @pytest.mark.parametrize("BufferClass", [ReplayBuffer, PrioritizedReplayBuffer, TreeBuffer])
    def test_buffer_initialization(self, BufferClass):
        """Test memory buffer initialization."""
        capacity = 100
        buffer = BufferClass(capacity=capacity)
        assert buffer.capacity == capacity
        assert len(buffer) == 0

    @pytest.mark.parametrize("BufferClass", [ReplayBuffer, PrioritizedReplayBuffer, TreeBuffer])
    def test_add_and_len(self, BufferClass):
        """Test adding samples and checking buffer length."""
        buffer = BufferClass(capacity=10)
        buffer.add({"obs": 1, "action": 0})
        assert len(buffer) == 1
        buffer.add({"obs": 2, "action": 1})
        assert len(buffer) == 2

    @pytest.mark.parametrize("BufferClass", [ReplayBuffer, PrioritizedReplayBuffer, TreeBuffer])
    def test_buffer_capacity(self, BufferClass):
        """Test that buffer does not exceed capacity."""
        capacity = 5
        buffer = BufferClass(capacity=capacity)
        for i in range(10):
            buffer.add({"obs": i})
        assert len(buffer) == capacity

    @pytest.mark.parametrize("BufferClass", [ReplayBuffer, PrioritizedReplayBuffer, TreeBuffer])
    def test_is_ready(self, BufferClass):
        """Test buffer readiness check."""
        buffer = BufferClass(capacity=20)
        assert not buffer.is_ready(min_size=10)
        for i in range(10):
            buffer.add({"obs": i})
        assert buffer.is_ready(min_size=10)
        assert not buffer.is_ready(min_size=11)

    @pytest.mark.parametrize("BufferClass", [ReplayBuffer, PrioritizedReplayBuffer, TreeBuffer])
    def test_clear_buffer(self, BufferClass):
        """Test clearing the buffer."""
        buffer = BufferClass(capacity=10)
        for i in range(5):
            buffer.add({"obs": i})
        buffer.clear()
        assert len(buffer) == 0

    @pytest.mark.parametrize("BufferClass", [ReplayBuffer, PrioritizedReplayBuffer, TreeBuffer])
    def test_sample_from_buffer(self, BufferClass):
        """Test sampling from the buffer."""
        buffer = BufferClass(capacity=20)
        for i in range(15):
            buffer.add({"id": i})

        sample = buffer.sample(batch_size=5)
        assert isinstance(sample, list)
        assert len(sample) == 5

        # Check for unique samples if possible
        if isinstance(buffer, ReplayBuffer):
            ids = {item["id"] for item in sample}
            assert len(ids) == 5


class TestSelfPlay:
    """Test suite for the SelfPlay class."""

    def test_self_play_initialization(self):
        """Test SelfPlay initialization."""
        config = Config()
        agent = Agent(config)
        memory_buffer = ReplayBuffer(capacity=100)
        self_play = SelfPlay(config, agent, memory_buffer)

        assert self_play.config == config
        assert self_play.agent == agent
        assert self_play.memory_buffer == memory_buffer

    def test_run_episode(self):
        """Test running a single self-play episode."""
        config = Config()
        config.action_space_size = 2
        config.observation_shape = (4,)
        agent = Agent(config)
        memory_buffer = ReplayBuffer(capacity=100)
        env = gym.make("CartPole-v1")
        self_play = SelfPlay(config, agent, memory_buffer)

        episode_data = self_play.run_episode(env)

        assert isinstance(episode_data, list)
        assert len(episode_data) > 0
        assert len(memory_buffer) == 1  # One episode was added

        # Check experience format
        for step in episode_data:
            assert "observation" in step
            assert "action" in step
            assert "reward" in step
            assert "policy" in step
            assert "done" in step

    def test_collect_samples(self):
        """Test collecting samples from multiple episodes."""
        config = Config()
        config.action_space_size = 2
        config.observation_shape = (4,)
        agent = Agent(config)
        memory_buffer = ReplayBuffer(capacity=100)
        env = gym.make("CartPole-v1")
        self_play = SelfPlay(config, agent, memory_buffer)

        num_episodes = 3
        self_play.collect_samples(env, num_episodes)
        assert len(memory_buffer) == num_episodes


if __name__ == "__main__":
    pytest.main([__file__])
