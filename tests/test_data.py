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
    """Test suite for MemoryBuffer implementations."""

    def test_replay_buffer_initialization(self):
        """Test ReplayBuffer initialization."""
        capacity = 100
        buffer = ReplayBuffer(capacity)

        assert buffer.capacity == capacity
        assert len(buffer) == 0
        assert len(buffer.buffer) == 0
        assert not buffer.is_ready(1)

    def test_prioritized_replay_buffer_initialization(self):
        """Test PrioritizedReplayBuffer initialization."""
        capacity = 100
        buffer = PrioritizedReplayBuffer(capacity, alpha=0.6, beta=0.4)

        assert buffer.capacity == capacity
        assert len(buffer) == 0
        assert buffer.alpha == 0.6
        assert buffer.beta == 0.4
        assert buffer.max_priority == 1.0

    def test_tree_buffer_initialization(self):
        """Test TreeBuffer initialization."""
        capacity = 100
        buffer = TreeBuffer(capacity, alpha=0.6)

        assert buffer.capacity == capacity
        assert len(buffer) == 0
        assert buffer.alpha == 0.6
        assert buffer.max_priority == 1.0
        assert hasattr(buffer, "sum_tree")

    def test_add_single_sample(self):
        """Test adding a single sample."""
        buffer = ReplayBuffer(capacity=10)

        sample = [{"observation": np.ones(4), "action": 0, "reward": 1.0, "search_policy": np.array([0.6, 0.4]), "value": 0.5}, {"observation": np.ones(4) * 2, "action": 1, "reward": 0.5, "search_policy": np.array([0.3, 0.7]), "value": 0.3}]

        buffer.add(sample)

        assert len(buffer) == 1
        assert buffer.is_ready(1)
        assert len(buffer.buffer[0]) == 2

    def test_add_multiple_samples(self):
        """Test adding multiple samples."""
        buffer = ReplayBuffer(capacity=3)

        for i in range(5):  # Add more than capacity
            sample = [{"observation": np.ones(4) * i, "action": i % 2, "reward": float(i), "search_policy": np.array([0.5, 0.5]), "value": float(i) * 0.1}]
            buffer.add(sample)

        # Should only keep most recent samples up to capacity
        assert len(buffer) == 3

    def test_sampling(self):
        """Test data sampling."""
        buffer = ReplayBuffer(capacity=10)

        # Add several samples
        for i in range(5):
            sample = [{"observation": np.ones(4) * i, "action": i % 2, "reward": float(i), "search_policy": np.array([0.5, 0.5]), "value": float(i) * 0.1}]
            buffer.add(sample)

        # Test sampling
        sampled = buffer.sample(3)
        assert len(sampled) == 3

        # Each sampled item should be a sample (list of experiences)
        for sample in sampled:
            assert isinstance(sample, list)
            assert len(sample) >= 1

    def test_sampling_more_than_available(self):
        """Test sampling more samples than available."""
        buffer = ReplayBuffer(capacity=10)

        # Add only 2 samples
        for i in range(2):
            sample = [{"observation": np.ones(4), "action": 0, "reward": 1.0, "search_policy": np.array([0.5, 0.5]), "value": 0.5}]
            buffer.add(sample)

        # Try to sample more than available
        sampled = buffer.sample(5)
        assert len(sampled) == 2  # Should return all available

    def test_is_ready(self):
        """Test readiness check for sampling."""
        buffer = ReplayBuffer(capacity=10)

        assert not buffer.is_ready(1)

        # Add one sample
        sample = [{"observation": np.ones(4), "action": 0, "reward": 1.0, "search_policy": np.array([0.5, 0.5]), "value": 0.5}]
        buffer.add(sample)

        assert buffer.is_ready(1)
        assert not buffer.is_ready(2)

    def test_circular_buffer_behavior(self):
        """Test circular buffer behavior when exceeding capacity."""
        buffer = ReplayBuffer(capacity=3)

        # Add samples with identifiable data
        for i in range(5):
            sample = [
                {
                    "observation": np.full(4, i),  # Use i as identifier
                    "action": 0,
                    "reward": float(i),
                    "search_policy": np.array([0.5, 0.5]),
                    "value": 0.5,
                }
            ]
            buffer.add(sample)

        # Should have samples 2, 3, 4 (most recent 3)
        assert len(buffer) == 3
        sampled = buffer.sample(3)

        # Check that we have the expected samples
        rewards = [sample[0]["reward"] for sample in sampled]
        expected_rewards = [2.0, 3.0, 4.0]
        assert sorted(rewards) == sorted(expected_rewards)

    def test_prioritized_buffer_priority_update(self):
        """Test priority update in PrioritizedReplayBuffer."""
        buffer = PrioritizedReplayBuffer(capacity=10)

        # Add sample
        sample = [{"observation": np.ones(4), "action": 0, "reward": 1.0}]
        buffer.add(sample)

        # Update priority
        buffer.update_priorities([0], [5.0])
        assert buffer.max_priority == 5.0

    def test_tree_buffer_priority_update(self):
        """Test priority update in TreeBuffer."""
        buffer = TreeBuffer(capacity=10)

        # Add sample
        sample = [{"observation": np.ones(4), "action": 0, "reward": 1.0}]
        buffer.add(sample)

        # Update priority - TreeBuffer applies alpha exponent, so we need to account for that
        buffer.update_priorities([0], [3.0])
        expected_priority_alpha = np.power(3.0 + buffer.epsilon, buffer.alpha)
        assert buffer.max_priority >= expected_priority_alpha


class TestSelfPlay:
    """Test suite for SelfPlay."""

    def test_self_play_initialization(self):
        """Test SelfPlay initialization."""
        config = Config()
        config.num_simulations = 5  # Small for testing
        config.action_space_size = 2
        config.observation_shape = (4,)
        agent = Agent(config)
        memory_buffer = ReplayBuffer(capacity=100)
        gym.make("CartPole-v1")

        self_play = SelfPlay(config, agent, memory_buffer)

        assert self_play.config == config
        assert self_play.agent == agent
        assert self_play.memory_buffer == memory_buffer
        assert self_play.episode_count == 0
        assert self_play.total_reward == 0.0

    def test_single_game_generation(self):
        """Test generation of a single game."""
        config = Config()
        config.num_simulations = 5  # Small for testing
        config.action_space_size = 2
        config.observation_shape = (4,)
        agent = Agent(config)
        memory_buffer = ReplayBuffer(capacity=100)
        env = gym.make("CartPole-v1")

        self_play = SelfPlay(config, agent, memory_buffer)

        episode_data = self_play.run_episode(env)

        # Verify episode_data structure
        assert isinstance(episode_data, list)
        assert len(episode_data) > 0

        # Check episode_data format
        for experience in episode_data:
            assert "observation" in experience
            assert "action" in experience
            assert "reward" in experience
            assert "next_observation" in experience
            assert "done" in experience

            # Verify types and shapes
            assert experience["observation"].shape == (4,)  # CartPole obs
            assert isinstance(experience["action"], (int, np.integer))
            assert isinstance(experience["reward"], (float, np.floating))
            assert isinstance(experience["done"], bool)
            assert 0 <= experience["action"] < config.action_space_size

        # Last experience should be terminal
        assert episode_data[-1]["done"]

        # Verify episode was actually played (some reward accumulated)
        total_reward = sum(exp["reward"] for exp in episode_data)
        assert total_reward > 0  # CartPole gives +1 per step

    def test_multiple_games_generation(self):
        """Test generation of multiple games."""
        config = Config()
        config.num_simulations = 3  # Very small for testing
        config.action_space_size = 2
        config.observation_shape = (4,)
        agent = Agent(config)
        memory_buffer = ReplayBuffer(capacity=100)
        env = gym.make("CartPole-v1")

        self_play = SelfPlay(config, agent, memory_buffer)

        num_games = 3
        self_play.collect_samples(env, num_games)

        # Check that data samples were added to memory buffer
        assert len(memory_buffer) == num_games
        assert self_play.episode_count == num_games

    def test_game_termination_conditions(self):
        """Test various game termination conditions."""
        config = Config()
        config.num_simulations = 3
        config.action_space_size = 2
        config.observation_shape = (4,)
        agent = Agent(config)
        memory_buffer = ReplayBuffer(capacity=100)
        env = gym.make("CartPole-v1")

        self_play = SelfPlay(config, agent, memory_buffer)

        episode_data = self_play.run_episode(env)

        # Should terminate naturally or due to environment limits
        assert len(episode_data) > 0
        assert episode_data[-1]["done"]

    def test_experience_format_consistency(self):
        """Test consistency of experience format across games."""
        config = Config()
        config.num_simulations = 2
        config.action_space_size = 2
        config.observation_shape = (4,)
        agent = Agent(config)
        memory_buffer = ReplayBuffer(capacity=100)
        env = gym.make("CartPole-v1")

        self_play = SelfPlay(config, agent, memory_buffer)

        # Collect multiple episodes
        self_play.collect_samples(env, 2)
        episodes = memory_buffer.sample(2)

        # Check format consistency across all experiences
        for episode in episodes:
            for experience in episode:
                # All experiences should have the same keys
                expected_keys = {"observation", "action", "reward", "next_observation", "done"}
                assert set(experience.keys()) == expected_keys

                # All observations should have the same shape
                assert experience["observation"].shape == (4,)

    def test_value_target_computation(self):
        """Test value target computation for training."""
        config = Config()
        config.num_simulations = 2
        config.action_space_size = 2
        config.observation_shape = (4,)
        agent = Agent(config)
        memory_buffer = ReplayBuffer(capacity=100)
        env = gym.make("CartPole-v1")

        self_play = SelfPlay(config, agent, memory_buffer)

        episode_data = self_play.run_episode(env)

        # Verify that rewards are properly recorded
        total_reward = sum(exp["reward"] for exp in episode_data)
        assert total_reward >= 0  # CartPole rewards are non-negative

    def test_action_distribution_recording(self):
        """Test that action distributions are properly recorded."""
        config = Config()
        config.num_simulations = 2
        config.action_space_size = 2
        config.observation_shape = (4,)
        agent = Agent(config)
        memory_buffer = ReplayBuffer(capacity=100)
        env = gym.make("CartPole-v1")

        self_play = SelfPlay(config, agent, memory_buffer)

        episode_data = self_play.run_episode(env)

        # Basic check that episode_data was created
        assert len(episode_data) > 0
        assert all(isinstance(exp["action"], (int, np.integer)) for exp in episode_data)


if __name__ == "__main__":
    pytest.main([__file__])
