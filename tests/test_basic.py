"""Basic integration tests for CuMind implementation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from cumind.agent.agent import Agent
from cumind.config import Config
from cumind.core.network import CuMindNetwork
from cumind.data.memory import Memory, MemoryBuffer, PrioritizedMemoryBuffer, TreeBuffer


def test_network_creation():
    """Test network creation and basic functionality."""
    config = Config()
    config.hidden_dim = 32
    config.num_blocks = 2
    config.action_space_size = 2
    config.observation_shape = (4,)

    key = jax.random.PRNGKey(42)
    rngs = nnx.Rngs(params=key)

    network = CuMindNetwork(observation_shape=config.observation_shape, action_space_size=config.action_space_size, hidden_dim=config.hidden_dim, num_blocks=config.num_blocks, conv_channels=config.conv_channels, rngs=rngs)

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

    network = CuMindNetwork(observation_shape=config.observation_shape, action_space_size=config.action_space_size, hidden_dim=config.hidden_dim, num_blocks=config.num_blocks, conv_channels=config.conv_channels, rngs=rngs)

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
    action, _ = agent.select_action(observation, training=False)

    assert isinstance(action, (int, np.integer))
    assert 0 <= action < config.action_space_size


def test_memory_buffer_functionality():
    """Test memory buffer operations for all buffer types."""
    # Test MemoryBuffer
    memory_buffer = MemoryBuffer(capacity=100)
    _test_buffer_operations(memory_buffer, "MemoryBuffer")

    # Test PrioritizedMemoryBuffer
    prioritized_buffer = PrioritizedMemoryBuffer(capacity=100)
    _test_buffer_operations(prioritized_buffer, "PrioritizedMemoryBuffer")

    # Test TreeBuffer
    tree_buffer = TreeBuffer(capacity=100)
    _test_buffer_operations(tree_buffer, "TreeBuffer")


def _test_buffer_operations(buffer: Memory, buffer_name: str):
    """Helper function to test buffer operations."""
    # Test empty buffer
    assert len(buffer) == 0, f"{buffer_name}: Empty buffer should have length 0"
    assert not buffer.is_ready(1), f"{buffer_name}: Empty buffer should not be ready"

    # Add dummy sample
    dummy_sample = [{"observation": np.ones(4), "action": 0, "reward": 1.0, "search_policy": np.array([0.6, 0.4]), "value": 0.5}, {"observation": np.ones(4), "action": 1, "reward": 0.5, "search_policy": np.array([0.3, 0.7]), "value": 0.3}]
    buffer.add(dummy_sample)

    assert len(buffer) == 1, f"{buffer_name}: Buffer should have 1 sample after adding"
    assert buffer.is_ready(1), f"{buffer_name}: Buffer should be ready after adding sample"

    # Sample from buffer
    sampled = buffer.sample(1)
    assert len(sampled) == 1, f"{buffer_name}: Should sample 1 item"
    assert len(sampled[0]) == 2, f"{buffer_name}: Sampled item should have 2 experiences"

    # Test priority update for prioritized buffers
    if hasattr(buffer, "update_priorities"):
        buffer.update_priorities([0], [2.0])


def test_prioritized_buffer_priority_update():
    """Test priority update functionality for prioritized buffers."""
    # Test PrioritizedMemoryBuffer priority update
    prioritized_buffer = PrioritizedMemoryBuffer(capacity=10)
    sample = [{"observation": np.ones(4), "action": 0, "reward": 1.0}]
    prioritized_buffer.add(sample)

    # Update priority
    prioritized_buffer.update_priorities([0], [5.0])
    assert prioritized_buffer.max_priority == 5.0

    # Test TreeBuffer priority update
    tree_buffer = TreeBuffer(capacity=10)
    tree_buffer.add(sample)

    # Update priority - TreeBuffer applies alpha exponent, so we need to account for that
    tree_buffer.update_priorities([0], [3.0])
    expected_priority_alpha = np.power(3.0 + tree_buffer.epsilon, tree_buffer.alpha)
    assert tree_buffer.max_priority >= expected_priority_alpha


def test_basic_integration():
    """Test basic integration of all components."""
    config = Config()
    config.num_simulations = 3  # Very small for testing
    config.action_space_size = 2
    config.observation_shape = (4,)

    # Create agent
    agent = Agent(config)

    # Test with different buffer types
    buffer_types = [MemoryBuffer, PrioritizedMemoryBuffer, TreeBuffer]

    for BufferClass in buffer_types:  # noqa: N806
        # Create memory buffer
        memory_buffer = BufferClass(capacity=10)

        # Generate some dummy experience
        observation = np.ones(4)
        action, _ = agent.select_action(observation, training=True)

        # Create experience entry
        experience = {"observation": observation, "action": action, "reward": 1.0, "search_policy": np.array([0.5, 0.5]), "value": 0.5}

        # Add to memory buffer
        memory_buffer.add([experience])

        # Verify integration
        assert len(memory_buffer) == 1, f"{BufferClass.__name__}: Should have 1 experience"
        assert 0 <= action < config.action_space_size


if __name__ == "__main__":
    pytest.main([__file__])
