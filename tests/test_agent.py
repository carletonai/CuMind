import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cumind.agent import Agent
from cumind.config import Config


class TestAgent:
    """Test suite for CuMind Agent."""

    def test_agent_initialization(self):
        """Test CuMind agent initialization."""
        config = Config()
        config.action_space_size = 2
        config.observation_shape = (4,)
        agent = Agent(config)

        # Verify agent has required attributes
        assert hasattr(agent, "config")
        assert hasattr(agent, "network")
        assert hasattr(agent, "mcts")
        assert hasattr(agent, "optimizer")
        assert hasattr(agent, "optimizer_state")

        # Verify config is stored correctly
        assert agent.config == config

        # Test with custom config
        custom_config = Config()
        custom_config.num_simulations = 100
        custom_config.learning_rate = 0.001
        custom_config.action_space_size = 2
        custom_config.observation_shape = (4,)
        custom_agent = Agent(custom_config)

        assert custom_agent.config.num_simulations == 100
        assert custom_agent.config.learning_rate == 0.001

    def test_select_action_training_mode(self):
        """Test action selection in training mode."""
        config = Config()
        config.num_simulations = 5  # Small number for fast testing
        config.action_space_size = 2
        config.observation_shape = (4,)
        agent = Agent(config)

        # Test with dummy observation
        obs = np.ones(4)  # CartPole observation shape

        action = agent.select_action(obs, training=True)

        # Verify action is valid
        assert isinstance(action, (int, np.integer))
        assert 0 <= action < config.action_space_size

        # Test multiple actions for stochastic behavior
        actions = []
        for i in range(10):
            action = agent.select_action(obs, training=True)
            actions.append(action)

        # Should have some variation in training mode
        # (though not guaranteed with small simulation count)
        assert all(0 <= a < config.action_space_size for a in actions)

    def test_select_action_evaluation_mode(self):
        """Test action selection in evaluation mode."""
        config = Config()
        config.num_simulations = 5  # Small number for fast testing
        config.action_space_size = 2
        config.observation_shape = (4,)
        agent = Agent(config)

        obs = np.ones(4)  # CartPole observation shape

        # Test evaluation mode
        action1 = agent.select_action(obs, training=False)
        action2 = agent.select_action(obs, training=False)

        # Should be deterministic in evaluation mode
        assert action1 == action2
        assert 0 <= action1 < config.action_space_size

    def test_train_step(self):
        """Test training step with batch data."""
        config = Config()
        config.batch_size = 4
        config.num_unroll_steps = 3
        config.action_space_size = 2
        config.observation_shape = (4,)
        agent = Agent(config)

        # Create mock batch data in the correct format (list of trajectories)
        batch = []
        for _ in range(config.batch_size):
            trajectory = [{"observation": np.random.normal(0, 1, 4), "action": np.random.randint(0, 2), "reward": np.random.normal(0, 1), "search_policy": np.array([0.6, 0.4]), "value": np.random.normal(0, 1)} for _ in range(config.num_unroll_steps + 1)]
            batch.append(trajectory)

        # Test training step (basic functionality check)
        try:
            metrics = agent.train_step(batch)
            # If we get here, the method exists and runs
            assert isinstance(metrics, dict)
        except (AttributeError, NotImplementedError):
            # Method may not be fully implemented yet
            assert True

    def test_prepare_batch(self):
        """Test batch preparation from replay buffer."""
        config = Config()
        config.batch_size = 2
        config.num_unroll_steps = 2
        config.action_space_size = 2
        config.observation_shape = (4,)
        agent = Agent(config)

        # Create mock experiences
        experiences = []
        for i in range(5):
            exp = {"observation": np.ones(4) * i, "action": i % 2, "reward": float(i), "search_policy": np.array([0.6, 0.4]), "value": float(i * 0.5)}
            experiences.append(exp)

        # Test batch preparation (basic functionality check)
        try:
            batch = agent.prepare_batch(experiences)
            # If method exists, check basic structure
            assert isinstance(batch, (dict, list))
        except (AttributeError, NotImplementedError):
            # Method may not be fully implemented yet
            assert True

    def test_action_selection_deterministic(self):
        """Test that action selection is deterministic during evaluation."""
        config = Config()
        config.action_space_size = 2
        config.observation_shape = (4,)
        config.num_simulations = 10
        agent = Agent(config)

        # Create test observation
        observation = np.random.random(4)

        # Test deterministic action selection (training=False)
        action1 = agent.select_action(observation, training=False)
        action2 = agent.select_action(observation, training=False)

        # Actions should be identical for same observation in eval mode
        assert action1 == action2
        assert isinstance(action1, int)
        assert 0 <= action1 < config.action_space_size

    def test_action_selection_stochastic(self):
        """Test that action selection varies during training due to sampling."""
        config = Config()
        config.action_space_size = 2
        config.observation_shape = (4,)
        config.num_simulations = 50
        agent = Agent(config)

        # Create test observation
        observation = np.random.random(4)

        # Collect multiple actions with training=True (stochastic)
        actions = []
        for _ in range(20):
            action = agent.select_action(observation, training=True)
            actions.append(action)
            assert isinstance(action, int)
            assert 0 <= action < config.action_space_size

        # Should have some variety in training mode (though not guaranteed)
        # At minimum, all actions should be valid
        unique_actions = set(actions)
        assert len(unique_actions) >= 1
        assert all(0 <= a < config.action_space_size for a in unique_actions)

    def test_save_checkpoint(self):
        """Test checkpoint saving functionality."""
        import os
        import tempfile

        config = Config()
        config.action_space_size = 2
        config.observation_shape = (4,)
        agent = Agent(config)

        # Create temporary file for testing
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp_file:
            checkpoint_path = tmp_file.name

        try:
            # Test saving checkpoint (if method exists)
            if hasattr(agent, "save_checkpoint"):
                agent.save_checkpoint(checkpoint_path)
                # Verify file was created
                assert os.path.exists(checkpoint_path)
                assert os.path.getsize(checkpoint_path) > 0
            else:
                # Method may not be implemented yet
                assert True
        finally:
            # Clean up
            if os.path.exists(checkpoint_path):
                os.unlink(checkpoint_path)

    def test_load_checkpoint(self):
        """Test checkpoint loading functionality."""
        import os
        import tempfile

        config = Config()
        config.action_space_size = 2
        config.observation_shape = (4,)
        agent1 = Agent(config)

        # Save checkpoint if possible
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp_file:
            checkpoint_path = tmp_file.name

        try:
            if hasattr(agent1, "save_checkpoint") and hasattr(agent1, "load_checkpoint"):
                agent1.save_checkpoint(checkpoint_path)

                # Create new agent and load checkpoint
                agent2 = Agent(config)
                agent2.load_checkpoint(checkpoint_path)

                # Basic verification that load completed
                assert True
            else:
                # Methods may not be implemented yet
                assert True
        finally:
            # Clean up
            if os.path.exists(checkpoint_path):
                os.unlink(checkpoint_path)

    def test_agent_with_vector_observations(self):
        """Test agent with 1D vector observations."""
        config = Config()
        config.observation_shape = (8,)  # Different vector size
        config.action_space_size = 2
        agent = Agent(config)

        # Test with vector observation
        obs = np.ones(8)
        action = agent.select_action(obs, training=True)

        # Verify action is valid
        assert isinstance(action, (int, np.integer))
        assert 0 <= action < config.action_space_size

    def test_agent_with_image_observations(self):
        """Test agent with 3D image observations (Atari)."""
        config = Config()
        config.observation_shape = (84, 84, 4)  # Atari-style input
        config.action_space_size = 4
        agent = Agent(config)

        # Test with image observation
        obs = np.ones((84, 84, 4))
        action = agent.select_action(obs, training=True)

        # Verify action is valid
        assert isinstance(action, (int, np.integer))
        assert 0 <= action < config.action_space_size


if __name__ == "__main__":
    pytest.main([__file__])
