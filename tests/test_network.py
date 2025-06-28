"""Tests for CuMindNetwork components and related neural network modules."""

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from cumind.core.network import (
    ConvEncoder,
    CuMindNetwork,
    DynamicsNetwork,
    PredictionNetwork,
    RepresentationNetwork,
    ResidualBlock,
    VectorEncoder,
)
from cumind.utils.prng import key


@pytest.fixture(autouse=True)
def reset_prng_manager_singleton():
    """Reset the key singleton before and after each test."""
    key.reset()
    yield
    key.reset()


class TestResidualBlock:
    """Test suite for ResidualBlock."""

    def test_residual_block_initialization(self):
        """Test ResidualBlock initialization."""
        key.seed(0)
        rngs = nnx.Rngs(params=key())

        channels = 32
        block = ResidualBlock(channels, rngs)

        # Test that block has required layers
        assert hasattr(block, "conv1")
        assert hasattr(block, "conv2")
        assert hasattr(block, "bn1")
        assert hasattr(block, "bn2")

        # Test with different channel counts
        for ch in [16, 64, 128]:
            key.seed(ch)
            rngs = nnx.Rngs(params=key())
            block = ResidualBlock(ch, rngs)
            assert hasattr(block, "conv1")

    def test_residual_block_forward(self):
        """Test ResidualBlock forward pass."""
        key.seed(0)
        rngs = nnx.Rngs(params=key())

        channels = 32
        block = ResidualBlock(channels, rngs)

        # Test forward pass
        batch_size, height, width = 2, 8, 8
        x = jnp.ones((batch_size, height, width, channels))
        output = block(x)

        # Output shape should match input shape
        assert output.shape == x.shape

        # Output should be different from input (not identity)
        assert not jnp.allclose(output, x)

    def test_residual_connection(self):
        """Test residual connection functionality."""
        key.seed(0)
        rngs = nnx.Rngs(params=key())

        channels = 16
        block = ResidualBlock(channels, rngs)

        batch_size, height, width = 1, 4, 4
        x = jax.random.normal(key(), (batch_size, height, width, channels))

        # Test forward pass
        output = block(x)

        # Check that output shape is preserved
        assert output.shape == x.shape

        # Check that output is different from input (transformation occurred)
        assert not jnp.allclose(output, x)


class TestVectorEncoder:
    """Test suite for VectorEncoder."""

    def test_vector_encoder_initialization(self):
        """Test VectorEncoder initialization."""
        key.seed(0)
        rngs = nnx.Rngs(params=key())

        observation_shape = (4,)
        hidden_dim = 64
        num_blocks = 2

        encoder = VectorEncoder(observation_shape, hidden_dim, num_blocks, rngs)

        # Test that encoder has required attributes
        assert hasattr(encoder, "layers")
        assert len(encoder.layers) == num_blocks
        assert encoder.hidden_dim == hidden_dim
        assert encoder.observation_shape == observation_shape

    def test_vector_encoder_forward(self):
        """Test VectorEncoder forward pass."""
        key.seed(0)
        rngs = nnx.Rngs(params=key())

        observation_shape = (4,)
        hidden_dim = 32
        num_blocks = 2
        batch_size = 3

        encoder = VectorEncoder(observation_shape, hidden_dim, num_blocks, rngs)

        # Test forward pass
        obs = jnp.ones((batch_size, observation_shape[0]))
        output = encoder(obs)

        # Check output shape
        assert output.shape == (batch_size, hidden_dim)

        # Test with different input sizes
        obs_large = jnp.ones((10, observation_shape[0]))
        output_large = encoder(obs_large)
        assert output_large.shape == (10, hidden_dim)

    def test_vector_encoder_gradients(self):
        """Test VectorEncoder gradient flow."""
        key.seed(0)
        rngs = nnx.Rngs(params=key())

        observation_shape = (8,)
        hidden_dim = 16
        num_blocks = 1

        encoder = VectorEncoder(observation_shape, hidden_dim, num_blocks, rngs)

        # Test forward pass
        obs = jax.random.normal(key(), (2, observation_shape[0]))
        output = encoder(obs)

        # Verify output shape
        assert output.shape == (2, hidden_dim)

        # Test that encoder has parameters
        state = nnx.state(encoder)
        assert len(state) > 0  # Should have parameters


class TestConvEncoder:
    """Test suite for ConvEncoder."""

    def test_conv_encoder_initialization(self):
        """Test ConvEncoder initialization."""
        key.seed(0)
        rngs = nnx.Rngs(params=key())
        observation_shape = (84, 84, 3)
        hidden_dim = 256
        num_blocks = 2
        conv_channels = 32

        encoder = ConvEncoder(observation_shape, hidden_dim, num_blocks, conv_channels, rngs)

        assert hasattr(encoder, "initial_conv")
        assert isinstance(encoder.initial_conv, nnx.Conv)
        assert hasattr(encoder, "residual_blocks")
        assert len(encoder.residual_blocks) == num_blocks
        assert isinstance(encoder.residual_blocks[0], ResidualBlock)
        assert hasattr(encoder, "final_dense")
        assert isinstance(encoder.final_dense, nnx.Linear)

    def test_conv_encoder_forward(self):
        """Test ConvEncoder forward pass."""
        key.seed(0)
        rngs = nnx.Rngs(params=key())
        observation_shape = (84, 84, 3)
        hidden_dim = 256
        num_blocks = 2
        conv_channels = 32
        batch_size = 4

        encoder = ConvEncoder(observation_shape, hidden_dim, num_blocks, conv_channels, rngs)
        obs = jnp.ones((batch_size, *observation_shape))
        output = encoder(obs)

        assert output.shape == (batch_size, hidden_dim)

    def test_conv_encoder_with_residual_blocks(self):
        """Test ConvEncoder with residual blocks."""
        key.seed(0)
        rngs = nnx.Rngs(params=key())
        observation_shape = (10, 10, 4)
        hidden_dim = 128
        num_blocks = 4
        conv_channels = 16
        batch_size = 2

        encoder = ConvEncoder(observation_shape, hidden_dim, num_blocks, conv_channels, rngs)
        obs = jnp.ones((batch_size, *observation_shape))
        output = encoder(obs)
        assert output.shape == (batch_size, hidden_dim)

        # Test gradient flow
        grad_fn = nnx.grad(lambda m, x: m(x).sum())
        grads = grad_fn(encoder, obs)
        assert grads is not None
        chex.assert_tree_all_finite(grads)


class TestRepresentationNetwork:
    """Test suite for RepresentationNetwork."""

    def test_representation_network_initialization(self):
        """Test RepresentationNetwork initialization."""
        key.seed(0)
        rngs = nnx.Rngs(params=key())
        hidden_dim = 64
        num_blocks = 2
        conv_channels = 16

        # Test with 1D observation shape
        net_1d = RepresentationNetwork((4,), hidden_dim, num_blocks, conv_channels, rngs)
        assert isinstance(net_1d.encoder, VectorEncoder)

        # Test with 3D observation shape
        net_3d = RepresentationNetwork((84, 84, 3), hidden_dim, num_blocks, conv_channels, rngs)
        assert isinstance(net_3d.encoder, ConvEncoder)

    def test_encoder_factory_method(self):
        """Test encoder factory method selection."""
        key.seed(0)
        rngs = nnx.Rngs(params=key())
        hidden_dim = 64
        num_blocks = 2
        conv_channels = 16

        # Create a dummy network to get access to the factory method
        net = RepresentationNetwork((1,), hidden_dim, num_blocks, conv_channels, rngs)

        # Test VectorEncoder selection for 1D observations
        net.observation_shape = (10,)
        encoder_1d = net._create_encoder(rngs)
        assert isinstance(encoder_1d, VectorEncoder)

        # Test ConvEncoder selection for 3D observations
        net.observation_shape = (10, 10, 1)
        encoder_3d = net._create_encoder(rngs)
        assert isinstance(encoder_3d, ConvEncoder)

        # Test error handling for unsupported shapes
        net.observation_shape = (10, 10)
        with pytest.raises(ValueError, match="Unsupported observation shape"):
            net._create_encoder(rngs)

    def test_representation_forward_1d(self):
        """Test RepresentationNetwork with 1D observations."""
        key.seed(0)
        rngs = nnx.Rngs(params=key())
        observation_shape = (8,)
        hidden_dim = 32
        net = RepresentationNetwork(observation_shape, hidden_dim, 2, 0, rngs)
        obs = jnp.ones((4, *observation_shape))
        hidden_state = net(obs)
        assert hidden_state.shape == (4, hidden_dim)

    def test_representation_forward_3d(self):
        """Test RepresentationNetwork with 3D observations."""
        key.seed(0)
        rngs = nnx.Rngs(params=key())
        observation_shape = (16, 16, 3)
        hidden_dim = 64
        net = RepresentationNetwork(observation_shape, hidden_dim, 2, 8, rngs)
        obs = jnp.ones((2, *observation_shape))
        hidden_state = net(obs)
        assert hidden_state.shape == (2, hidden_dim)


class TestDynamicsNetwork:
    """Test suite for DynamicsNetwork."""

    def test_dynamics_network_initialization(self):
        """Test DynamicsNetwork initialization."""
        key.seed(0)
        rngs = nnx.Rngs(params=key())
        hidden_dim = 64
        action_space_size = 5
        num_blocks = 3
        net = DynamicsNetwork(hidden_dim, action_space_size, num_blocks, rngs)

        assert isinstance(net.action_embedding, nnx.Embed)
        assert net.action_embedding.num_embeddings == action_space_size
        assert len(net.layers) == num_blocks
        assert isinstance(net.reward_head, nnx.Linear)

    def test_dynamics_forward(self):
        """Test DynamicsNetwork forward pass."""
        key.seed(0)
        rngs = nnx.Rngs(params=key())
        hidden_dim = 32
        action_space_size = 4
        num_blocks = 2
        batch_size = 8
        net = DynamicsNetwork(hidden_dim, action_space_size, num_blocks, rngs)
        state = jnp.ones((batch_size, hidden_dim))
        action = jnp.zeros(batch_size, dtype=jnp.int32)
        next_state, reward = net(state, action)

        assert next_state.shape == (batch_size, hidden_dim)
        assert reward.shape == (batch_size, 1)

    def test_action_embedding(self):
        """Test action embedding in dynamics network."""
        key.seed(0)
        rngs = nnx.Rngs(params=key())
        hidden_dim = 16
        action_space_size = 7
        batch_size = 3
        net = DynamicsNetwork(hidden_dim, action_space_size, 1, rngs)
        action = jnp.array([0, 2, 6])
        action_embedded = net.action_embedding(action)

        assert action_embedded.shape == (batch_size, hidden_dim)

    def test_reward_prediction(self):
        """Test reward prediction accuracy."""
        key.seed(0)
        rngs = nnx.Rngs(params=key())
        hidden_dim = 32
        action_space_size = 4
        num_blocks = 2
        batch_size = 8
        net = DynamicsNetwork(hidden_dim, action_space_size, num_blocks, rngs)
        state = jnp.ones((batch_size, hidden_dim))
        action = jnp.zeros(batch_size, dtype=jnp.int32)
        _, reward = net(state, action)

        assert reward.shape == (batch_size, 1)

        grad_fn = nnx.grad(lambda m, s, a: m(s, a)[1].sum())
        grads = grad_fn(net, state, action)
        assert grads is not None
        chex.assert_tree_all_finite(grads)


class TestPredictionNetwork:
    """Test suite for PredictionNetwork."""

    def test_prediction_network_initialization(self):
        """Test PredictionNetwork initialization."""
        key.seed(0)
        rngs = nnx.Rngs(params=key())
        hidden_dim = 128
        action_space_size = 10
        net = PredictionNetwork(hidden_dim, action_space_size, rngs)

        assert isinstance(net.policy_head, nnx.Linear)
        assert isinstance(net.value_head, nnx.Linear)
        assert net.policy_head.out_features == action_space_size
        assert net.value_head.out_features == 1

    def test_prediction_forward(self):
        """Test PredictionNetwork forward pass."""
        key.seed(0)
        rngs = nnx.Rngs(params=key())
        hidden_dim = 64
        action_space_size = 5
        batch_size = 4
        net = PredictionNetwork(hidden_dim, action_space_size, rngs)
        hidden_state = jnp.ones((batch_size, hidden_dim))
        policy_logits, value = net(hidden_state)

        assert policy_logits.shape == (batch_size, action_space_size)
        assert value.shape == (batch_size, 1)

    def test_policy_head(self):
        """Test policy head functionality."""
        key.seed(0)
        rngs = nnx.Rngs(params=key())
        hidden_dim = 64
        action_space_size = 5
        batch_size = 4
        net = PredictionNetwork(hidden_dim, action_space_size, rngs)
        hidden_state = jnp.ones((batch_size, hidden_dim))
        policy_logits, _ = net(hidden_state)

        assert policy_logits.shape == (batch_size, action_space_size)

    def test_value_head(self):
        """Test value head functionality."""
        key.seed(0)
        rngs = nnx.Rngs(params=key())
        hidden_dim = 64
        action_space_size = 5
        batch_size = 4
        net = PredictionNetwork(hidden_dim, action_space_size, rngs)
        hidden_state = jnp.ones((batch_size, hidden_dim))
        _, value = net(hidden_state)

        assert value.shape == (batch_size, 1)


class TestCuMindNetwork:
    """Test suite for complete CuMindNetwork."""

    @pytest.fixture
    def setup_1d(self):
        key.seed(0)
        return {
            "observation_shape": (4,),
            "action_space_size": 2,
            "hidden_dim": 32,
            "num_blocks": 1,
            "conv_channels": 0,  # Not used
            "rngs": nnx.Rngs(params=key()),
        }

    @pytest.fixture
    def setup_3d(self):
        key.seed(1)
        return {
            "observation_shape": (16, 16, 3),
            "action_space_size": 5,
            "hidden_dim": 64,
            "num_blocks": 2,
            "conv_channels": 8,
            "rngs": nnx.Rngs(params=key()),
        }

    def test_cumind_network_initialization(self, setup_1d):
        """Test CuMindNetwork initialization."""
        net = CuMindNetwork(**setup_1d)
        assert isinstance(net.representation_network, RepresentationNetwork)
        assert isinstance(net.dynamics_network, DynamicsNetwork)
        assert isinstance(net.prediction_network, PredictionNetwork)
        assert isinstance(net.representation_network.encoder, VectorEncoder)

    def test_initial_inference(self, setup_1d):
        """Test initial inference step."""
        net = CuMindNetwork(**setup_1d)
        batch_size = 4
        obs = jnp.ones((batch_size, *setup_1d["observation_shape"]))
        hidden_state, policy_logits, value = net.initial_inference(obs)

        assert hidden_state.shape == (batch_size, setup_1d["hidden_dim"])
        assert policy_logits.shape == (batch_size, setup_1d["action_space_size"])
        assert value.shape == (batch_size, 1)

    def test_recurrent_inference(self, setup_1d):
        """Test recurrent inference step."""
        net = CuMindNetwork(**setup_1d)
        batch_size = 4
        hidden_state = jnp.ones((batch_size, setup_1d["hidden_dim"]))
        action = jnp.zeros(batch_size, dtype=jnp.int32)
        next_state, reward, policy_logits, value = net.recurrent_inference(hidden_state, action)

        assert next_state.shape == (batch_size, setup_1d["hidden_dim"])
        assert reward.shape == (batch_size, 1)
        assert policy_logits.shape == (batch_size, setup_1d["action_space_size"])
        assert value.shape == (batch_size, 1)

    def test_network_integration(self, setup_1d):
        """Test integration between network components."""
        net = CuMindNetwork(**setup_1d)
        batch_size = 2
        hidden_state = jnp.ones((batch_size, setup_1d["hidden_dim"]))
        action = jnp.ones(batch_size, dtype=jnp.int32)

        def loss_fn(model, state, act):
            next_hs, reward, pl, val = model.recurrent_inference(state, act)
            return (next_hs.sum() + reward.sum() + pl.sum() + val.sum()), reward

        grad_fn = nnx.grad(loss_fn, has_aux=True)
        grads, reward = grad_fn(net, hidden_state, action)

        assert reward.shape == (batch_size, 1)
        assert grads is not None
        chex.assert_tree_all_finite(grads)

    def test_cumind_network_with_1d_observations(self, setup_1d):
        """Test CuMindNetwork with vector observations."""
        net = CuMindNetwork(**setup_1d)
        batch_size = 2
        obs = jnp.ones((batch_size, *setup_1d["observation_shape"]))

        hidden_state, policy_logits, value = net.initial_inference(obs)
        assert isinstance(net.representation_network.encoder, VectorEncoder)
        assert hidden_state.shape == (batch_size, setup_1d["hidden_dim"])

        action = jnp.argmax(policy_logits, axis=-1)
        next_state, reward, _, _ = net.recurrent_inference(hidden_state, action)
        assert next_state.shape == (batch_size, setup_1d["hidden_dim"])
        assert reward.shape == (batch_size, 1)

    def test_cumind_network_with_3d_observations(self, setup_3d):
        """Test CuMindNetwork with image observations."""
        net = CuMindNetwork(**setup_3d)
        batch_size = 2
        obs = jnp.ones((batch_size, *setup_3d["observation_shape"]))

        hidden_state, policy_logits, value = net.initial_inference(obs)
        assert isinstance(net.representation_network.encoder, ConvEncoder)
        assert hidden_state.shape == (batch_size, setup_3d["hidden_dim"])

        action = jnp.argmax(policy_logits, axis=-1)
        next_state, reward, _, _ = net.recurrent_inference(hidden_state, action)
        assert next_state.shape == (batch_size, setup_3d["hidden_dim"])
        assert reward.shape == (batch_size, 1)

    def test_unsupported_observation_shapes(self, setup_1d):
        """Test error handling for unsupported observation shapes."""
        params = setup_1d.copy()
        with pytest.raises(ValueError, match="Unsupported observation shape"):
            params["observation_shape"] = (10, 10)
            CuMindNetwork(**params)

        with pytest.raises(ValueError, match="Unsupported observation shape"):
            params["observation_shape"] = (1, 2, 3, 4)
            CuMindNetwork(**params)


if __name__ == "__main__":
    pytest.main([__file__])
