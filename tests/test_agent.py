"""Tests for the Agent class verifying initialization, action selection, state persistence, and input handling."""

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from cumind.agent.agent import Agent
from cumind.config import Config


class TestAgent:
    """Test suite for the Agent class."""

    def test_agent_initialization(self):
        """Test agent initialization."""
        config = Config()
        config.env_action_space_size = 2
        config.env_observation_shape = (4,)
        agent = Agent(config)

        assert agent.config == config
        assert hasattr(agent, "network")
        assert hasattr(agent, "optimizer")
        assert hasattr(agent, "optimizer_state")
        assert hasattr(agent, "mcts")

    def test_select_action_training_mode(self):
        """Test action selection in training mode."""
        config = Config()
        config.mcts_num_simulations = 5
        config.env_action_space_size = 2
        config.env_observation_shape = (4,)
        agent = Agent(config)

        obs = np.ones(4)
        action, policy = agent.select_action(obs, training=True)

        assert isinstance(action, (int, np.integer))
        assert isinstance(policy, np.ndarray)
        assert len(policy) == config.env_action_space_size
        assert np.isclose(np.sum(policy), 1.0)

    def test_select_action_evaluation_mode(self):
        """Test action selection in evaluation mode is deterministic."""
        config = Config()
        config.mcts_num_simulations = 5
        config.env_action_space_size = 2
        config.env_observation_shape = (4,)
        agent = Agent(config)

        obs = np.ones(4)
        action1, _ = agent.select_action(obs, training=False)
        action2, _ = agent.select_action(obs, training=False)

        assert action1 == action2

    def test_save_and_load_state(self):
        """Test saving and loading agent state."""
        config = Config()
        config.env_action_space_size = 2
        config.env_observation_shape = (4,)
        agent1 = Agent(config)
        obs = np.ones(4)
        agent1.select_action(obs)  # Run one step to have state

        state = agent1.save_state()

        agent2 = Agent(config)
        agent2.load_state(state)

        # Check that network parameters are the same
        params1 = nnx.state(agent1.network, nnx.Param)
        params2 = nnx.state(agent2.network, nnx.Param)

        chex.assert_trees_all_close(params1, params2, rtol=1e-6)

    def test_agent_with_vector_observations(self):
        """Test agent with 1D vector observations."""
        config = Config()
        config.env_observation_shape = (8,)
        config.env_action_space_size = 2
        agent = Agent(config)

        obs = np.ones(8)
        action, _ = agent.select_action(obs, training=True)
        assert isinstance(action, (int, np.integer))

    def test_agent_with_image_observations(self):
        """Test agent with 3D image observations (Atari)."""
        config = Config()
        config.env_observation_shape = (84, 84, 4)
        config.env_action_space_size = 4
        agent = Agent(config)

        obs = np.ones((84, 84, 4))
        action, _ = agent.select_action(obs, training=True)
        assert isinstance(action, (int, np.integer))


if __name__ == "__main__":
    pytest.main([__file__])
