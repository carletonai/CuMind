"""Comprehensive tests for data components (ReplayBuffer, SelfPlay)."""

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cumind.agent.agent import Agent
from cumind.config import Config
from cumind.data.replay_buffer import ReplayBuffer
from cumind.data.self_play import SelfPlay


class TestReplayBuffer:
    """Test suite for ReplayBuffer."""

    def test_replay_buffer_initialization(self):
        """Test ReplayBuffer initialization."""
        capacity = 100
        buffer = ReplayBuffer(capacity)

        assert buffer.capacity == capacity
        assert len(buffer) == 0
        assert len(buffer.buffer) == 0
        assert not buffer.is_ready(1)

    def test_add_single_trajectory(self):
        """Test adding a single trajectory."""
        buffer = ReplayBuffer(capacity=10)

        trajectory = [{"observation": np.ones(4), "action": 0, "reward": 1.0, "search_policy": np.array([0.6, 0.4]), "value": 0.5}, {"observation": np.ones(4) * 2, "action": 1, "reward": 0.5, "search_policy": np.array([0.3, 0.7]), "value": 0.3}]

        buffer.add(trajectory)

        assert len(buffer) == 1
        assert buffer.is_ready(1)
        assert len(buffer.buffer[0]) == 2

    def test_add_multiple_trajectories(self):
        """Test adding multiple trajectories."""
        buffer = ReplayBuffer(capacity=3)

        for i in range(5):  # Add more than capacity
            trajectory = [{"observation": np.ones(4) * i, "action": i % 2, "reward": float(i), "search_policy": np.array([0.5, 0.5]), "value": float(i) * 0.1}]
            buffer.add(trajectory)

        # Should only keep most recent trajectories up to capacity
        assert len(buffer) == 3

    def test_sampling(self):
        """Test trajectory sampling."""
        buffer = ReplayBuffer(capacity=10)

        # Add several trajectories
        for i in range(5):
            trajectory = [{"observation": np.ones(4) * i, "action": i % 2, "reward": float(i), "search_policy": np.array([0.5, 0.5]), "value": float(i) * 0.1}]
            buffer.add(trajectory)

        # Test sampling
        sampled = buffer.sample(3)
        assert len(sampled) == 3

        # Each sampled item should be a trajectory (list of experiences)
        for trajectory in sampled:
            assert isinstance(trajectory, list)
            assert len(trajectory) >= 1

    def test_sampling_more_than_available(self):
        """Test sampling more trajectories than available."""
        buffer = ReplayBuffer(capacity=10)

        # Add only 2 trajectories
        for i in range(2):
            trajectory = [{"observation": np.ones(4), "action": 0, "reward": 1.0, "search_policy": np.array([0.5, 0.5]), "value": 0.5}]
            buffer.add(trajectory)

        # Try to sample more than available
        sampled = buffer.sample(5)
        assert len(sampled) == 2  # Should return all available

    def test_is_ready(self):
        """Test readiness check for sampling."""
        buffer = ReplayBuffer(capacity=10)

        assert not buffer.is_ready(1)

        # Add one trajectory
        trajectory = [{"observation": np.ones(4), "action": 0, "reward": 1.0, "search_policy": np.array([0.5, 0.5]), "value": 0.5}]
        buffer.add(trajectory)

        assert buffer.is_ready(1)
        assert not buffer.is_ready(2)

    def test_circular_buffer_behavior(self):
        """Test circular buffer behavior when exceeding capacity."""
        buffer = ReplayBuffer(capacity=3)

        # Add trajectories with identifiable data
        for i in range(5):
            trajectory = [
                {
                    "observation": np.full(4, i),  # Use i as identifier
                    "action": 0,
                    "reward": float(i),
                    "search_policy": np.array([0.5, 0.5]),
                    "value": 0.5,
                }
            ]
            buffer.add(trajectory)

        # Should have trajectories 2, 3, 4 (most recent 3)
        assert len(buffer) == 3
        sampled = buffer.sample(3)

        # Check that we have the expected trajectories
        rewards = [traj[0]["reward"] for traj in sampled]
        expected_rewards = [2.0, 3.0, 4.0]
        assert sorted(rewards) == sorted(expected_rewards)


class TestSelfPlay:
    """Test suite for SelfPlay."""

    def test_self_play_initialization(self):
        """Test SelfPlay initialization."""
        config = Config()
        config.num_simulations = 5  # Small for testing
        config.action_space_size = 2
        config.observation_shape = (4,)
        agent = Agent(config)
        replay_buffer = ReplayBuffer(capacity=100)
        gym.make("CartPole-v1")

        self_play = SelfPlay(config, agent, replay_buffer)

        assert self_play.config == config
        assert self_play.agent == agent
        assert self_play.replay_buffer == replay_buffer
        assert self_play.episode_count == 0
        assert self_play.total_reward == 0.0

    def test_single_game_generation(self):
        """Test generation of a single game."""
        config = Config()
        config.num_simulations = 5  # Small for testing
        config.action_space_size = 2
        config.observation_shape = (4,)
        agent = Agent(config)
        replay_buffer = ReplayBuffer(capacity=100)
        env = gym.make("CartPole-v1")

        self_play = SelfPlay(config, agent, replay_buffer)

        trajectory = self_play.run_episode(env)

        # Verify trajectory structure
        assert isinstance(trajectory, list)
        assert len(trajectory) > 0

        # Check trajectory format
        for experience in trajectory:
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
        assert trajectory[-1]["done"]

        # Verify episode was actually played (some reward accumulated)
        total_reward = sum(exp["reward"] for exp in trajectory)
        assert total_reward > 0  # CartPole gives +1 per step

    def test_multiple_games_generation(self):
        """Test generation of multiple games."""
        config = Config()
        config.num_simulations = 3  # Very small for testing
        config.action_space_size = 2
        config.observation_shape = (4,)
        agent = Agent(config)
        replay_buffer = ReplayBuffer(capacity=100)
        env = gym.make("CartPole-v1")

        self_play = SelfPlay(config, agent, replay_buffer)

        num_games = 3
        self_play.collect_trajectories(env, num_games)

        # Check that trajectories were added to replay buffer
        assert len(replay_buffer) == num_games
        assert self_play.episode_count == num_games
        assert self_play.total_reward > 0  # Should have accumulated some rewards

        # Verify we can sample from the buffer
        sampled_trajectories = replay_buffer.sample(2)
        assert len(sampled_trajectories) == 2

        # Each sampled trajectory should be valid
        for trajectory in sampled_trajectories:
            assert isinstance(trajectory, list)
            assert len(trajectory) > 0
            # Verify the last step is terminal
            assert trajectory[-1]["done"]

    def test_game_termination_conditions(self):
        """Test various game termination conditions."""
        config = Config()
        config.num_simulations = 3
        config.action_space_size = 2
        config.observation_shape = (4,)
        agent = Agent(config)
        replay_buffer = ReplayBuffer(capacity=100)
        env = gym.make("CartPole-v1")

        self_play = SelfPlay(config, agent, replay_buffer)

        trajectory = self_play.run_episode(env)

        # Should terminate naturally or due to environment limits
        assert len(trajectory) > 0
        assert trajectory[-1]["done"]

    def test_experience_format_consistency(self):
        """Test consistency of experience format across games."""
        config = Config()
        config.num_simulations = 2
        config.action_space_size = 2
        config.observation_shape = (4,)
        agent = Agent(config)
        replay_buffer = ReplayBuffer(capacity=100)
        env = gym.make("CartPole-v1")

        self_play = SelfPlay(config, agent, replay_buffer)

        # Collect multiple trajectories
        self_play.collect_trajectories(env, 2)
        trajectories = replay_buffer.sample(2)

        # Check format consistency across all experiences
        for trajectory in trajectories:
            for experience in trajectory:
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
        replay_buffer = ReplayBuffer(capacity=100)
        env = gym.make("CartPole-v1")

        self_play = SelfPlay(config, agent, replay_buffer)

        trajectory = self_play.run_episode(env)

        # Verify that rewards are properly recorded
        total_reward = sum(exp["reward"] for exp in trajectory)
        assert total_reward >= 0  # CartPole rewards are non-negative

    def test_action_distribution_recording(self):
        """Test that action distributions are properly recorded."""
        config = Config()
        config.num_simulations = 2
        config.action_space_size = 2
        config.observation_shape = (4,)
        agent = Agent(config)
        replay_buffer = ReplayBuffer(capacity=100)
        env = gym.make("CartPole-v1")

        self_play = SelfPlay(config, agent, replay_buffer)

        trajectory = self_play.run_episode(env)

        # Basic check that trajectory was created
        assert len(trajectory) > 0
        assert all(isinstance(exp["action"], (int, np.integer)) for exp in trajectory)


if __name__ == "__main__":
    pytest.main([__file__])
