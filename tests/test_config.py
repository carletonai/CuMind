"""Test suite for CuMind configuration system."""

from unittest.mock import patch

import pytest

from cumind.core.resnet import ResNet
from cumind.utils.config import (
    Configuration,
    DynamicsConfig,
    HotSwappableConfig,
    MemoryConfig,
    PredictionConfig,
    RepresentationConfig,
    cfg,
)
from cumind.utils.prng import key


class TestHotSwappableConfig:
    """Test the base HotSwappableConfig class."""

    def test_hot_swappable_config_init(self):
        """Test initialization of HotSwappableConfig."""
        config = HotSwappableConfig(type="test.module.Class")
        assert config.type == "test.module.Class"

    def test_hot_swappable_config_none_type(self):
        """Test that None type raises error on call."""
        config = HotSwappableConfig(type=None)
        with pytest.raises(ValueError, match="Type must be specified"):
            config()

    def test_hot_swappable_config_extras_default(self):
        """Test that extras() returns empty dict by default."""
        config = HotSwappableConfig(type="test.module.Class")
        assert config.extras() == {}


class TestRepresentationConfig:
    """Test RepresentationConfig functionality."""

    def test_representation_config_init(self):
        """Test RepresentationConfig initialization."""
        config = RepresentationConfig()
        assert config.type == "cumind.core.resnet.ResNet"
        assert config.num_blocks == 2
        assert config.conv_channels == 32

    def test_representation_config_extras(self):
        """Test RepresentationConfig extras method."""
        key.seed(cfg.seed)
        config = RepresentationConfig()
        extras = config.extras()

        assert "input_shape" in extras
        assert "rngs" in extras
        assert extras["input_shape"] == cfg.env.observation_shape

    def test_representation_config_instantiation(self):
        """Test that RepresentationConfig can instantiate ResNet."""
        key.seed(cfg.seed)
        config = RepresentationConfig()
        network = config()

        # Use type() instead of isinstance() for nnx modules
        assert type(network).__name__ == "ResNet"
        assert network.num_blocks == 2
        assert network.conv_channels == 32
        assert network.hidden_dim == 128

    def test_representation_config_custom_params(self):
        """Test RepresentationConfig with custom parameters."""
        config = RepresentationConfig(num_blocks=5, conv_channels=64)
        key.seed(cfg.seed)
        network = config()

        assert network.num_blocks == 5
        assert network.conv_channels == 64


class TestDynamicsConfig:
    """Test DynamicsConfig functionality."""

    def test_dynamics_config_init(self):
        """Test DynamicsConfig initialization."""
        config = DynamicsConfig()
        assert config.type == "cumind.core.mlp.MLPWithEmbedding"
        assert config.num_blocks == 2

    def test_dynamics_config_extras(self):
        """Test DynamicsConfig extras method."""
        config = DynamicsConfig()
        extras = config.extras()

        assert "hidden_dim" in extras
        assert "embedding_size" in extras
        assert "rngs" in extras
        assert extras["hidden_dim"] == cfg.networks.hidden_dim
        assert extras["embedding_size"] == cfg.env.action_space_size

    def test_dynamics_config_instantiation(self):
        """Test that DynamicsConfig can instantiate MLPWithEmbedding."""
        key.seed(cfg.seed)
        config = DynamicsConfig()
        network = config()

        # Use type() instead of isinstance() for nnx modules
        assert type(network).__name__ == "MLPWithEmbedding"
        assert network.num_blocks == 2
        assert network.hidden_dim == cfg.networks.hidden_dim
        assert network.embedding_size == cfg.env.action_space_size

    def test_dynamics_config_custom_params(self):
        """Test DynamicsConfig with custom parameters."""
        config = DynamicsConfig(num_blocks=4)
        key.seed(cfg.seed)
        network = config()

        assert network.num_blocks == 4


class TestPredictionConfig:
    """Test PredictionConfig functionality."""

    def test_prediction_config_init(self):
        """Test PredictionConfig initialization."""
        config = PredictionConfig()
        assert config.type == "cumind.core.mlp.MLPDual"

    def test_prediction_config_extras(self):
        """Test PredictionConfig extras method."""
        config = PredictionConfig()
        extras = config.extras()

        assert "hidden_dim" in extras
        assert "output_size" in extras
        assert "rngs" in extras
        assert extras["hidden_dim"] == cfg.networks.hidden_dim
        assert extras["output_size"] == cfg.env.action_space_size

    def test_prediction_config_instantiation(self):
        """Test that PredictionConfig can instantiate MLPDual."""
        key.seed(cfg.seed)
        config = PredictionConfig()
        network = config()

        # Use type() instead of isinstance() for nnx modules
        assert type(network).__name__ == "MLPDual"
        assert network.hidden_dim == cfg.networks.hidden_dim
        assert network.output_size == cfg.env.action_space_size


class TestMemoryConfig:
    """Test MemoryConfig functionality."""

    def test_memory_config_init(self):
        """Test MemoryConfig initialization."""
        config = MemoryConfig()
        assert config.type == "cumind.data.memory.MemoryBuffer"
        assert config.capacity == 2000
        assert config.min_size == 100
        assert config.min_pct == 0.1

    def test_memory_config_extras(self):
        """Test MemoryConfig extras method."""
        config = MemoryConfig()
        extras = config.extras()

        # MemoryConfig extras should be empty since dataclass fields are handled by base class
        for item in extras:
            print(item)
        assert extras == {}

    def test_memory_config_instantiation(self):
        """Test that MemoryConfig can instantiate MemoryBuffer."""
        config = MemoryConfig()
        memory = config()

        # Use type() instead of isinstance() for consistency
        assert type(memory).__name__ == "MemoryBuffer"
        assert memory.capacity == 2000

    def test_memory_config_custom_params(self):
        """Test MemoryConfig with custom parameters."""
        config = MemoryConfig(capacity=5000, min_size=200)
        memory = config()

        assert memory.capacity == 5000
        # min_size is not passed to constructor, only used for configuration


class TestConfigurationIntegration:
    """Test integration of all config components."""

    def test_configuration_singleton(self):
        """Test that cfg is a singleton instance."""
        assert isinstance(cfg, Configuration)
        # Use the global cfg instance directly
        assert cfg is cfg

    def test_configuration_hot_swappable_modules(self):
        """Test that all hot-swappable modules are properly configured."""
        assert isinstance(cfg.representation, RepresentationConfig)
        assert isinstance(cfg.dynamics, DynamicsConfig)
        assert isinstance(cfg.prediction, PredictionConfig)
        assert isinstance(cfg.memory, MemoryConfig)

    def test_network_construction_integration(self):
        """Test complete network construction using config."""
        key.seed(cfg.seed)

        # Test that all networks can be constructed
        representation = cfg.representation()
        dynamics = cfg.dynamics()
        prediction = cfg.prediction()
        memory = cfg.memory()

        # Use type() instead of isinstance() for nnx modules
        assert type(representation).__name__ == "ResNet"
        assert type(dynamics).__name__ == "MLPWithEmbedding"
        assert type(prediction).__name__ == "MLPDual"
        assert type(memory).__name__ == "MemoryBuffer"

    def test_configuration_validation(self):
        """Test that configuration validation works."""
        # This should not raise any errors
        cfg._validate()


class TestDynamicResolution:
    """Test dynamic module resolution functionality."""

    def test_string_type_resolution(self):
        """Test that string types are properly resolved."""
        config = RepresentationConfig(type="cumind.core.resnet.ResNet")
        key.seed(cfg.seed)
        network = config()
        assert type(network).__name__ == "ResNet"

    def test_class_type_direct_usage(self):
        """Test that class types can be used directly."""
        config = RepresentationConfig(type=ResNet)
        key.seed(cfg.seed)
        network = config()
        assert type(network).__name__ == "ResNet"

    def test_resolve_not_called_for_class_types(self):
        """Test that resolve is not called for class types."""
        with patch("cumind.utils.resolve.resolve") as mock_resolve:
            config = RepresentationConfig(type=ResNet)
            key.seed(cfg.seed)
            config()
            mock_resolve.assert_not_called()


class TestParameterFiltering:
    """Test parameter filtering functionality."""

    def test_parameter_filtering_removes_unused(self):
        """Test that unused parameters are filtered out."""

        # Create a mock class with specific constructor signature
        class MockClass:
            def __init__(self, param1, param2):
                self.param1 = param1
                self.param2 = param2

        # Create a custom config class that inherits from RepresentationConfig
        class CustomConfig(RepresentationConfig):
            def extras(self):
                return {"param1": "value1", "param2": "value2", "unused_param": "should_be_filtered"}

        config = CustomConfig(type=MockClass)
        instance = config()
        assert hasattr(instance, "param1")
        assert hasattr(instance, "param2")
        assert not hasattr(instance, "unused_param")

    def test_self_parameter_filtered(self):
        """Test that 'self' parameter is always filtered out."""

        class MockClass:
            def __init__(self, param1):
                self.param1 = param1

        # Create a custom config class that inherits from RepresentationConfig
        class CustomConfig(RepresentationConfig):
            def extras(self):
                return {"param1": "value1", "self": "should_be_filtered"}

        config = CustomConfig(type=MockClass)
        instance = config()
        assert hasattr(instance, "param1")
        assert not hasattr(instance, "self")


class TestErrorHandling:
    """Test error handling in configuration system."""

    def test_invalid_type_string(self):
        """Test error handling for invalid type strings."""
        config = RepresentationConfig(type="invalid.module.Class")
        key.seed(cfg.seed)
        with pytest.raises(ImportError):
            config()

    def test_missing_required_parameters(self):
        """Test error handling for missing required parameters."""

        class MockClass:
            def __init__(self, required_param):
                self.required_param = required_param

        # Create a custom config class that inherits from RepresentationConfig
        class CustomConfig(RepresentationConfig):
            def extras(self):
                return {}  # No parameters provided

        config = CustomConfig(type=MockClass)
        with pytest.raises(TypeError):
            config()

    def test_none_type_error(self):
        """Test error handling for None type."""
        config = HotSwappableConfig(type=None)
        with pytest.raises(ValueError, match="Type must be specified"):
            config()


if __name__ == "__main__":
    pytest.main([__file__])
