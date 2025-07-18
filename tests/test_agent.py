"""Tests for the Agent class verifying initialization, action selection, state persistence, and input handling."""

import chex
import numpy as np
import pytest
from flax import nnx

from cumind.agent.agent import Agent
from cumind.utils.config import cfg


class TestAgent:
    """Test suite for the Agent class."""

    def test_agent_initialization(self):
        """Test agent initialization."""
        agent = Agent()

        assert hasattr(agent, "network")
        assert hasattr(agent, "optimizer")
        assert hasattr(agent, "optimizer_state")
        assert hasattr(agent, "mcts")

    def test_select_action_training_mode(self):
        """Test action selection in training mode."""
        agent = Agent()

        obs = np.ones(4)
        action, policy = agent.select_action(obs, training=True)

        assert isinstance(action, (int, np.integer))
        assert isinstance(policy, np.ndarray)
        assert len(policy) == cfg.env.action_space_size
        assert np.isclose(np.sum(policy), 1.0)

    def test_select_action_evaluation_mode(self):
        """Test action selection in evaluation mode is deterministic."""
        agent = Agent()

        obs = np.ones(4)
        action1, _ = agent.select_action(obs, training=False)
        action2, _ = agent.select_action(obs, training=False)

        assert action1 == action2

    def test_save_and_load_state(self):
        """Test saving and loading agent state."""
        agent1 = Agent()
        obs = np.ones(4)
        agent1.select_action(obs)  # Run one step to have state

        state = agent1.save_state()

        agent2 = Agent()
        agent2.load_state(state)

        # Check that network parameters are the same
        params1 = nnx.state(agent1.network, nnx.Param)
        params2 = nnx.state(agent2.network, nnx.Param)

        chex.assert_trees_all_close(params1, params2, rtol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__])
