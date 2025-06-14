"""Tests for MuZero implementation."""

import numpy as np
import pytest
import torch

from muzero import MuZeroAgent, MuZeroConfig, MuZeroNetwork


def test_config_creation():
    """Test MuZero configuration creation."""
    config = MuZeroConfig()
    assert config.hidden_dim == 64
    assert config.action_space_size == 4
    assert config.num_simulations == 50


def test_network_creation():
    """Test MuZero network creation."""
    config = MuZeroConfig()
    network = MuZeroNetwork(
        observation_shape=config.observation_shape,
        action_space_size=config.action_space_size,
        hidden_dim=config.hidden_dim,
    )

    # Test initial inference
    obs = torch.randn(1, *config.observation_shape)
    hidden_state, policy_logits, value = network.initial_inference(obs)

    assert hidden_state.shape[0] == 1
    assert policy_logits.shape == (1, config.action_space_size)
    assert value.shape == (1, 1)

    # Test recurrent inference
    action = torch.randint(0, config.action_space_size, (1,))
    next_hidden_state, reward, policy_logits, value = network.recurrent_inference(
        hidden_state, action
    )

    assert next_hidden_state.shape == hidden_state.shape
    assert reward.shape == (1, 1)
    assert policy_logits.shape == (1, config.action_space_size)
    assert value.shape == (1, 1)


def test_agent_creation():
    """Test MuZero agent creation."""
    config = MuZeroConfig(observation_shape=(4, 84, 84))  # Different format for testing
    agent = MuZeroAgent(config)

    # Test action selection
    observation = np.random.randn(*config.observation_shape)
    action = agent.select_action(observation, training=False)

    assert isinstance(action, (int, np.integer))
    assert 0 <= action < config.action_space_size


def test_modular_encoder_1d():
    """Test VectorEncoder for 1D observations."""
    config = MuZeroConfig(observation_shape=(4,), action_space_size=2)  # CartPole-like
    network = MuZeroNetwork(
        observation_shape=config.observation_shape,
        action_space_size=config.action_space_size,
        hidden_dim=config.hidden_dim,
    )

    # Test with 1D observation
    obs = torch.randn(1, 4)  # Batch size 1, 4 features
    hidden_state, policy_logits, value = network.initial_inference(obs)

    assert hidden_state.shape[0] == 1
    assert hidden_state.shape[1] == config.hidden_dim
    assert policy_logits.shape == (1, config.action_space_size)
    assert value.shape == (1, 1)


def test_modular_encoder_3d():
    """Test ConvEncoder for 3D observations."""
    config = MuZeroConfig(observation_shape=(3, 84, 84), action_space_size=4)  # Atari-like
    network = MuZeroNetwork(
        observation_shape=config.observation_shape,
        action_space_size=config.action_space_size,
        hidden_dim=config.hidden_dim,
    )

    # Test with 3D observation
    obs = torch.randn(1, 3, 84, 84)  # Batch size 1, 3 channels, 84x84
    hidden_state, policy_logits, value = network.initial_inference(obs)

    assert hidden_state.shape[0] == 1
    assert hidden_state.shape[1] == config.hidden_dim
    assert policy_logits.shape == (1, config.action_space_size)
    assert value.shape == (1, 1)


def test_unsupported_observation_shape():
    """Test that unsupported observation shapes raise appropriate errors."""
    with pytest.raises(ValueError, match="Unsupported observation shape"):
        # 2D observations are not supported (yet)
        MuZeroNetwork(
            observation_shape=(84, 84),  # Missing channel dimension
            action_space_size=4,
            hidden_dim=64,
        )


def test_training_step():
    """Test training step."""
    config = MuZeroConfig(batch_size=2)
    agent = MuZeroAgent(config)

    # Dummy batch
    batch = []
    losses = agent.train_step(batch)

    assert "value_loss" in losses
    assert "policy_loss" in losses
    assert "reward_loss" in losses
    assert all(isinstance(loss, float) for loss in losses.values())


if __name__ == "__main__":
    pytest.main([__file__])
