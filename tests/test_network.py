"""Tests for CuMindNetwork components and related neural network modules."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from cumind.core.blocks import ResidualBlock
from cumind.core.encoder import ConvEncoder, VectorEncoder
from cumind.core.mlp import MLPDual, MLPWithEmbedding
from cumind.core.network import CuMindNetwork
from cumind.core.resnet import ResNet
from cumind.utils.config import cfg
from cumind.utils.prng import key


@pytest.fixture(autouse=True)
def reset_prng_manager_singleton():
    """Reset the key singleton before and after each test."""
    key.seed(42)  # Initialize with a default seed
    yield


class TestResidualBlock:
    """Test suite for ResidualBlock."""

    def test_residual_block_initialization(self):
        """Test ResidualBlock initialization."""
        key.seed(0)
        rngs = nnx.Rngs(params=key.get())

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
            rngs = nnx.Rngs(params=key.get())
            block = ResidualBlock(ch, rngs)
            assert hasattr(block, "conv1")

    def test_residual_block_forward(self):
        """Test ResidualBlock forward pass."""
        key.seed(0)
        rngs = nnx.Rngs(params=key.get())

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
        rngs = nnx.Rngs(params=key.get())

        channels = 16
        block = ResidualBlock(channels, rngs)

        batch_size, height, width = 1, 4, 4
        x = jax.random.normal(key.get(), (batch_size, height, width, channels))

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
        rngs = nnx.Rngs(params=key.get())

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
        rngs = nnx.Rngs(params=key.get())

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
        rngs = nnx.Rngs(params=key.get())

        observation_shape = (8,)
        hidden_dim = 16
        num_blocks = 1

        encoder = VectorEncoder(observation_shape, hidden_dim, num_blocks, rngs)

        # Test forward pass
        obs = jax.random.normal(key.get(), (2, observation_shape[0]))
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
        rngs = nnx.Rngs(params=key.get())
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
        rngs = nnx.Rngs(params=key.get())
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
        rngs = nnx.Rngs(params=key.get())
        observation_shape = (10, 10, 4)
        hidden_dim = 128
        num_blocks = 4
        conv_channels = 16

        encoder = ConvEncoder(observation_shape, hidden_dim, num_blocks, conv_channels, rngs)

        # Test forward pass
        batch_size = 2
        obs = jax.random.normal(key.get(), (batch_size, *observation_shape))
        output = encoder(obs)

        # Verify output shape
        assert output.shape == (batch_size, hidden_dim)

        # Test that encoder has parameters
        state = nnx.state(encoder)
        assert len(state) > 0  # Should have parameters


class TestResNet:
    """Test suite for ResNet."""

    def test_resnet_initialization(self):
        """Test ResNet initialization."""
        key.seed(0)
        rngs = nnx.Rngs(params=key.get())

        observation_shape = (4,)
        hidden_dim = 64
        num_blocks = 2
        conv_channels = 16

        resnet = ResNet(hidden_dim, observation_shape, num_blocks, conv_channels, rngs)

        assert hasattr(resnet, "encoder")
        assert resnet.hidden_dim == hidden_dim
        assert resnet.input_shape == observation_shape

    def test_resnet_forward_1d(self):
        """Test ResNet forward pass with 1D observations."""
        key.seed(0)
        rngs = nnx.Rngs(params=key.get())

        observation_shape = (8,)
        hidden_dim = 32
        num_blocks = 1
        conv_channels = 8

        resnet = ResNet(hidden_dim, observation_shape, num_blocks, conv_channels, rngs)

        # Test forward pass
        batch_size = 3
        obs = jnp.ones((batch_size, observation_shape[0]))
        output = resnet(obs)

        assert output.shape == (batch_size, hidden_dim)

    def test_resnet_forward_3d(self):
        """Test ResNet forward pass with 3D observations."""
        key.seed(0)
        rngs = nnx.Rngs(params=key.get())

        observation_shape = (10, 10, 3)
        hidden_dim = 64
        num_blocks = 2
        conv_channels = 16

        resnet = ResNet(hidden_dim, observation_shape, num_blocks, conv_channels, rngs)

        # Test forward pass
        batch_size = 2
        obs = jnp.ones((batch_size, *observation_shape))
        output = resnet(obs)

        assert output.shape == (batch_size, hidden_dim)


class TestMLPWithEmbedding:
    """Test suite for MLPWithEmbedding."""

    def test_mlp_with_embedding_initialization(self):
        """Test MLPWithEmbedding initialization."""
        key.seed(0)
        rngs = nnx.Rngs(params=key.get())

        hidden_dim = 64
        embedding_size = 4
        num_blocks = 2

        mlp = MLPWithEmbedding(hidden_dim, embedding_size, num_blocks, rngs)

        assert hasattr(mlp, "embedding")
        assert hasattr(mlp, "layers")
        assert mlp.hidden_dim == hidden_dim
        assert mlp.embedding_size == embedding_size

    def test_mlp_with_embedding_forward(self):
        """Test MLPWithEmbedding forward pass."""
        key.seed(0)
        rngs = nnx.Rngs(params=key.get())

        hidden_dim = 32
        embedding_size = 3
        num_blocks = 1

        mlp = MLPWithEmbedding(hidden_dim, embedding_size, num_blocks, rngs)

        # Test forward pass
        batch_size = 2
        hidden_state = jnp.ones((batch_size, hidden_dim))
        actions = jnp.array([0, 1])

        next_state, reward = mlp(hidden_state, actions)

        assert next_state.shape == (batch_size, hidden_dim)
        assert reward.shape == (batch_size, 1)

    def test_action_embedding(self):
        """Test action embedding functionality."""
        key.seed(0)
        rngs = nnx.Rngs(params=key.get())

        hidden_dim = 16
        embedding_size = 2
        num_blocks = 1

        mlp = MLPWithEmbedding(hidden_dim, embedding_size, num_blocks, rngs)

        # Test with different actions
        batch_size = 3
        hidden_state = jnp.ones((batch_size, hidden_dim))

        for action in range(embedding_size):
            actions = jnp.array([action] * batch_size)
            next_state, reward = mlp(hidden_state, actions)

            assert next_state.shape == (batch_size, hidden_dim)
            assert reward.shape == (batch_size, 1)

    def test_reward_prediction(self):
        """Test reward prediction functionality."""
        key.seed(0)
        rngs = nnx.Rngs(params=key.get())

        hidden_dim = 8
        embedding_size = 2
        num_blocks = 1

        mlp = MLPWithEmbedding(hidden_dim, embedding_size, num_blocks, rngs)

        # Test forward pass
        batch_size = 2
        hidden_state = jax.random.normal(key.get(), (batch_size, hidden_dim))
        actions = jnp.array([0, 1])

        next_state, reward = mlp(hidden_state, actions)

        # Verify outputs
        assert next_state.shape == (batch_size, hidden_dim)
        assert reward.shape == (batch_size, 1)

        # Test that outputs are different from inputs
        assert not jnp.allclose(next_state, hidden_state)


class TestMLPDual:
    """Test suite for MLPDual."""

    def test_mlp_dual_initialization(self):
        """Test MLPDual initialization."""
        key.seed(0)
        rngs = nnx.Rngs(params=key.get())

        hidden_dim = 64
        output_size = 4

        mlp = MLPDual(hidden_dim, output_size, rngs)

        assert hasattr(mlp, "head1")
        assert hasattr(mlp, "head2")
        assert mlp.hidden_dim == hidden_dim
        assert mlp.output_size == output_size

    def test_mlp_dual_forward(self):
        """Test MLPDual forward pass."""
        key.seed(0)
        rngs = nnx.Rngs(params=key.get())

        hidden_dim = 32
        output_size = 3

        mlp = MLPDual(hidden_dim, output_size, rngs)

        # Test forward pass
        batch_size = 2
        hidden_state = jnp.ones((batch_size, hidden_dim))

        policy, value = mlp(hidden_state)

        assert policy.shape == (batch_size, output_size)
        assert value.shape == (batch_size, 1)

    def test_policy_head(self):
        """Test policy head functionality."""
        key.seed(0)
        rngs = nnx.Rngs(params=key.get())

        hidden_dim = 16
        output_size = 2

        mlp = MLPDual(hidden_dim, output_size, rngs)

        # Test policy head
        batch_size = 3
        hidden_state = jnp.ones((batch_size, hidden_dim))

        policy, _ = mlp(hidden_state)

        # Verify policy shape (these are raw logits, not normalized probabilities)
        assert policy.shape == (batch_size, output_size)
        # Don't check for normalization since these are raw logits

    def test_value_head(self):
        """Test value head functionality."""
        key.seed(0)
        rngs = nnx.Rngs(params=key.get())

        hidden_dim = 8
        output_size = 2

        mlp = MLPDual(hidden_dim, output_size, rngs)

        # Test value head
        batch_size = 2
        hidden_state = jax.random.normal(key.get(), (batch_size, hidden_dim))

        _, value = mlp(hidden_state)

        # Verify value shape
        assert value.shape == (batch_size, 1)


class TestCuMindNetwork:
    """Test suite for CuMindNetwork."""

    @pytest.fixture
    def setup_1d(self):
        """Setup for 1D observation tests."""
        key.seed(cfg.seed)
        rngs = nnx.Rngs(params=key.get())
        repre_net = cfg.representation()
        dyna_net = cfg.dynamics()
        pred_net = cfg.prediction()
        network = CuMindNetwork(repre_net, dyna_net, pred_net)
        return network, rngs

    @pytest.fixture
    def setup_3d(self):
        """Setup for 3D observation tests."""
        key.seed(cfg.seed)
        rngs = nnx.Rngs(params=key.get())

        # Create networks directly with 3D observation shape
        from cumind.core.mlp import MLPDual, MLPWithEmbedding
        from cumind.core.resnet import ResNet

        repre_net = ResNet(
            hidden_dim=cfg.networks.hidden_dim,
            input_shape=(10, 10, 3),  # 3D observation shape
            num_blocks=cfg.representation.num_blocks,
            conv_channels=cfg.representation.conv_channels,
            rngs=rngs,
        )
        dyna_net = MLPWithEmbedding(hidden_dim=cfg.networks.hidden_dim, embedding_size=cfg.env.action_space_size, num_blocks=cfg.dynamics.num_blocks, rngs=rngs)
        pred_net = MLPDual(hidden_dim=cfg.networks.hidden_dim, output_size=cfg.env.action_space_size, rngs=rngs)
        network = CuMindNetwork(repre_net, dyna_net, pred_net)
        return network, rngs

    def test_cumind_network_initialization(self, setup_1d):
        """Test CuMindNetwork initialization."""
        network, _ = setup_1d

        assert hasattr(network, "representation_network")
        assert hasattr(network, "dynamics_network")
        assert hasattr(network, "prediction_network")

    def test_initial_inference(self, setup_1d):
        """Test initial inference functionality."""
        network, _ = setup_1d

        batch_size = 2
        obs = jnp.ones((batch_size, 4))

        hidden_state, policy_logits, value = network.initial_inference(obs)

        assert hidden_state.shape == (batch_size, cfg.networks.hidden_dim)
        assert policy_logits.shape == (batch_size, cfg.env.action_space_size)
        assert value.shape == (batch_size, 1)

    def test_recurrent_inference(self, setup_1d):
        """Test recurrent inference functionality."""
        network, _ = setup_1d

        batch_size = 2
        hidden_state = jnp.ones((batch_size, cfg.networks.hidden_dim))
        actions = jnp.array([0, 1])

        next_state, reward, next_policy, next_value = network.recurrent_inference(hidden_state, actions)

        assert next_state.shape == (batch_size, cfg.networks.hidden_dim)
        assert reward.shape == (batch_size, 1)
        assert next_policy.shape == (batch_size, cfg.env.action_space_size)
        assert next_value.shape == (batch_size, 1)

    def test_network_integration(self, setup_1d):
        """Test complete network integration."""
        network, _ = setup_1d

        batch_size = 2
        obs = jnp.ones((batch_size, 4))

        # Test initial inference
        hidden_state, policy_logits, value = network.initial_inference(obs)

        # Test recurrent inference
        actions = jnp.array([0, 1])
        next_state, reward, next_policy, next_value = network.recurrent_inference(hidden_state, actions)

        # Verify all outputs have correct shapes
        assert hidden_state.shape == (batch_size, cfg.networks.hidden_dim)
        assert policy_logits.shape == (batch_size, cfg.env.action_space_size)
        assert value.shape == (batch_size, 1)
        assert next_state.shape == (batch_size, cfg.networks.hidden_dim)
        assert reward.shape == (batch_size, 1)
        assert next_policy.shape == (batch_size, cfg.env.action_space_size)
        assert next_value.shape == (batch_size, 1)

    def test_cumind_network_with_1d_observations(self, setup_1d):
        """Test CuMindNetwork with 1D observations."""
        network, _ = setup_1d

        batch_size = 3
        obs = jnp.ones((batch_size, 4))

        # Test initial inference
        hidden_state, policy_logits, value = network.initial_inference(obs)

        # Verify shapes
        assert hidden_state.shape == (batch_size, cfg.networks.hidden_dim)
        assert policy_logits.shape == (batch_size, cfg.env.action_space_size)
        assert value.shape == (batch_size, 1)

        # Test recurrent inference
        actions = jnp.array([0, 1, 0])
        next_state, reward, next_policy, next_value = network.recurrent_inference(hidden_state, actions)

        # Verify shapes
        assert next_state.shape == (batch_size, cfg.networks.hidden_dim)
        assert reward.shape == (batch_size, 1)
        assert next_policy.shape == (batch_size, cfg.env.action_space_size)
        assert next_value.shape == (batch_size, 1)

    def test_cumind_network_with_3d_observations(self, setup_3d):
        """Test CuMindNetwork with 3D observations."""
        network, _ = setup_3d

        batch_size = 2
        obs = jnp.ones((batch_size, 10, 10, 3))

        # Test initial inference
        hidden_state, policy_logits, value = network.initial_inference(obs)

        # Verify shapes
        assert hidden_state.shape == (batch_size, cfg.networks.hidden_dim)
        assert policy_logits.shape == (batch_size, cfg.env.action_space_size)
        assert value.shape == (batch_size, 1)

        # Test recurrent inference
        actions = jnp.array([0, 1])
        next_state, reward, next_policy, next_value = network.recurrent_inference(hidden_state, actions)

        # Verify shapes
        assert next_state.shape == (batch_size, cfg.networks.hidden_dim)
        assert reward.shape == (batch_size, 1)
        assert next_policy.shape == (batch_size, cfg.env.action_space_size)
        assert next_value.shape == (batch_size, 1)


if __name__ == "__main__":
    pytest.main([__file__])
