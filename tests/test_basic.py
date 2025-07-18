"""Basic integration tests for CuMind implementation."""

import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from cumind.agent.agent import Agent
from cumind.core.network import CuMindNetwork
from cumind.data.memory import Memory, MemoryBuffer, PrioritizedMemoryBuffer, TreeBuffer
from cumind.utils.config import cfg
from cumind.utils.prng import key


@pytest.fixture(autouse=True)
def reset_prng_manager_singleton():
    """Reset the PRNGManager singleton before and after each test."""
    key.reset()
    yield
    key.reset()


def test_network_creation():
    """Test network creation and basic functionality."""
    key.seed(42)
    rngs = nnx.Rngs(params=key())

    repre_net = cfg.representation()
    dyna_net = cfg.dynamics()
    pred_net = cfg.prediction()
    network = CuMindNetwork(repre_net, dyna_net, pred_net, rngs)

    assert hasattr(network, "representation_network")
    assert hasattr(network, "dynamics_network")
    assert hasattr(network, "prediction_network")


def test_network_inference():
    """Test network inference functionality."""
    key.seed(42)
    rngs = nnx.Rngs(params=key())

    repre_net = cfg.representation()
    dyna_net = cfg.dynamics()
    pred_net = cfg.prediction()
    network = CuMindNetwork(repre_net, dyna_net, pred_net, rngs)

    # Test initial inference
    batch_size = 2
    obs = jnp.ones((batch_size, 4))
    hidden_state, policy_logits, value = network.initial_inference(obs)

    assert hidden_state.shape == (batch_size, cfg.networks.hidden_dim)
    assert policy_logits.shape == (batch_size, cfg.env.action_space_size)
    assert value.shape == (batch_size, 1)

    # Test recurrent inference
    actions = jnp.array([0, 1])
    next_state, reward, next_policy, next_value = network.recurrent_inference(hidden_state, actions)

    assert next_state.shape == (batch_size, cfg.networks.hidden_dim)
    assert reward.shape == (batch_size, 1)
    assert next_policy.shape == (batch_size, cfg.env.action_space_size)
    assert next_value.shape == (batch_size, 1)


def test_agent_creation_and_action_selection():
    """Test agent creation and action selection."""
    agent = Agent()

    # Test agent has required components
    assert hasattr(agent, "network")
    assert hasattr(agent, "mcts")
    assert hasattr(agent, "optimizer")
    assert hasattr(agent, "optimizer_state")

    # Test action selection
    observation = np.ones(4)
    action, _ = agent.select_action(observation, training=False)

    assert isinstance(action, (int, np.integer))
    assert 0 <= action < cfg.env.action_space_size


def test_memory_buffer_functionality():
    """Test memory buffer operations for all buffer types."""
    # Test MemoryBuffer
    memory_buffer = cfg.memory()
    _test_buffer_operations(memory_buffer, "MemoryBuffer")

    # Test PrioritizedMemoryBuffer
    prioritized_buffer = PrioritizedMemoryBuffer(capacity=100, alpha=cfg.memory.per_alpha, epsilon=cfg.memory.per_epsilon, beta=cfg.memory.per_beta)
    _test_buffer_operations(prioritized_buffer, "PrioritizedMemoryBuffer")

    # Test TreeBuffer
    tree_buffer = TreeBuffer(capacity=100, alpha=cfg.memory.per_alpha, epsilon=cfg.memory.per_epsilon)
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
    result = buffer.sample(1)
    if isinstance(result, tuple):
        sampled, _ = result
    else:
        sampled = result
    assert len(sampled) == 1, f"{buffer_name}: Should sample 1 item"
    assert len(sampled[0]) == 2, f"{buffer_name}: Sampled item should have 2 experiences"

    # Test priority update for prioritized buffers
    if isinstance(buffer, (PrioritizedMemoryBuffer, TreeBuffer)):
        buffer.update_priorities([0], [2.0])


def test_prioritized_buffer_priority_update():
    """Test priority update functionality for prioritized buffers."""
    # Test PrioritizedMemoryBuffer priority update
    prioritized_buffer = PrioritizedMemoryBuffer(capacity=10, alpha=cfg.memory.per_alpha, epsilon=cfg.memory.per_epsilon, beta=cfg.memory.per_beta)
    sample = [{"observation": np.ones(4), "action": 0, "reward": 1.0}]
    prioritized_buffer.add(sample)

    # Update priority
    prioritized_buffer.update_priorities([0], [5.0])
    assert prioritized_buffer.max_priority == 5.0

    # Test TreeBuffer priority update
    tree_buffer = TreeBuffer(capacity=10, alpha=cfg.memory.per_alpha, epsilon=cfg.memory.per_epsilon)
    tree_buffer.add(sample)

    # Update priority - TreeBuffer applies alpha exponent, so we need to account for that
    tree_buffer.update_priorities([0], [3.0])
    expected_priority_alpha = np.power(3.0 + tree_buffer.epsilon, tree_buffer.alpha)
    assert tree_buffer.max_priority >= expected_priority_alpha


def test_basic_integration():
    """Test basic integration of all components."""
    # Create agent
    agent = Agent()

    # Test with different buffer types
    buffer_types = [
        (MemoryBuffer, {"capacity": 10}),
        (PrioritizedMemoryBuffer, {"capacity": 10, "alpha": cfg.memory.per_alpha, "epsilon": cfg.memory.per_epsilon, "beta": cfg.memory.per_beta}),
        (TreeBuffer, {"capacity": 10, "alpha": cfg.memory.per_alpha, "epsilon": cfg.memory.per_epsilon}),
    ]

    for BufferClass, kwargs in buffer_types:  # noqa: N806
        # Create memory buffer
        memory_buffer = BufferClass(**kwargs)

        # Generate some dummy experience
        observation = np.ones(4)
        action, _ = agent.select_action(observation, training=True)

        # Create experience entry
        experience = {"observation": observation, "action": action, "reward": 1.0, "search_policy": np.array([0.5, 0.5]), "value": 0.5}

        # Add to memory buffer
        memory_buffer.add([experience])

        # Verify integration
        assert len(memory_buffer) == 1, f"{BufferClass.__name__}: Should have 1 experience"
        assert 0 <= action < cfg.env.action_space_size


if __name__ == "__main__":
    pytest.main([__file__])
