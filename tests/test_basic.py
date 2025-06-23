"""Basic integration tests for CuMind implementation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from cumind.agent.agent import Agent
from cumind.config import Config
from cumind.core.network import CuMindNetwork
from cumind.data.replay_buffer import ReplayBuffer


def test_network_creation():
    """Test network creation and basic functionality."""
    config = Config()
    config.hidden_dim = 32
    config.num_blocks = 2
    config.action_space_size = 2
    config.observation_shape = (4,)

    key = jax.random.PRNGKey(42)
    rngs = nnx.Rngs(params=key)

    network = CuMindNetwork(observation_shape=config.observation_shape, action_space_size=config.action_space_size, hidden_dim=config.hidden_dim, num_blocks=config.num_blocks, rngs=rngs)

    assert hasattr(network, "representation_network")
    assert hasattr(network, "dynamics_network")
    assert hasattr(network, "prediction_network")


def test_network_inference():
    """Test network inference functionality."""
    config = Config()
    config.hidden_dim = 32
    config.num_blocks = 2
    config.action_space_size = 2
    config.observation_shape = (4,)

    key = jax.random.PRNGKey(42)
    rngs = nnx.Rngs(params=key)

    network = CuMindNetwork(observation_shape=config.observation_shape, action_space_size=config.action_space_size, hidden_dim=config.hidden_dim, num_blocks=config.num_blocks, rngs=rngs)

    # Test initial inference
    batch_size = 2
    obs = jnp.ones((batch_size, 4))
    hidden_state, policy_logits, value = network.initial_inference(obs)

    assert hidden_state.shape == (batch_size, config.hidden_dim)
    assert policy_logits.shape == (batch_size, config.action_space_size)
    assert value.shape == (batch_size, 1)

    # Test recurrent inference
    actions = jnp.array([0, 1])
    next_state, reward, next_policy, next_value = network.recurrent_inference(hidden_state, actions)

    assert next_state.shape == (batch_size, config.hidden_dim)
    assert reward.shape == (batch_size, 1)
    assert next_policy.shape == (batch_size, config.action_space_size)
    assert next_value.shape == (batch_size, 1)


def test_agent_creation_and_action_selection():
    """Test agent creation and action selection."""
    config = Config()
    config.num_simulations = 5  # Small for testing
    config.action_space_size = 2
    config.observation_shape = (4,)

    agent = Agent(config)

    # Test agent has required components
    assert hasattr(agent, "network")
    assert hasattr(agent, "mcts")
    assert hasattr(agent, "optimizer")
    assert hasattr(agent, "optimizer_state")

    # Test action selection
    observation = np.ones(4)
    action = agent.select_action(observation, training=False)

    assert isinstance(action, (int, np.integer))
    assert 0 <= action < config.action_space_size


def test_replay_buffer_functionality():
    """Test replay buffer operations."""
    replay_buffer = ReplayBuffer(capacity=100)

    # Test empty buffer
    assert len(replay_buffer) == 0
    assert not replay_buffer.is_ready(1)

    # Add dummy trajectory
    dummy_trajectory = [{"observation": np.ones(4), "action": 0, "reward": 1.0, "search_policy": np.array([0.6, 0.4]), "value": 0.5}, {"observation": np.ones(4), "action": 1, "reward": 0.5, "search_policy": np.array([0.3, 0.7]), "value": 0.3}]
    replay_buffer.add(dummy_trajectory)

    assert len(replay_buffer) == 1
    assert replay_buffer.is_ready(1)

    # Sample from buffer
    sampled = replay_buffer.sample(1)
    assert len(sampled) == 1
    assert len(sampled[0]) == 2  # Two experiences in trajectory


def test_basic_integration():
    """Test basic integration of all components."""
    config = Config()
    config.num_simulations = 3  # Very small for testing
    config.action_space_size = 2
    config.observation_shape = (4,)

    # Create agent
    agent = Agent(config)

    # Create replay buffer
    replay_buffer = ReplayBuffer(capacity=10)

    # Generate some dummy experience
    observation = np.ones(4)
    action = agent.select_action(observation, training=True)

    # Create experience entry
    experience = {"observation": observation, "action": action, "reward": 1.0, "search_policy": np.array([0.5, 0.5]), "value": 0.5}

    # Add to replay buffer
    replay_buffer.add([experience])

    # Verify integration
    assert len(replay_buffer) == 1
    assert 0 <= action < config.action_space_size


if __name__ == "__main__":
    pytest.main([__file__])
