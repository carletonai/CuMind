"""Test suite for CuMind configuration system."""

import os
import tempfile
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


@pytest.fixture(autouse=True)
def reset_config():
    """Reset the configuration singleton before and after each test."""
    # Reset the singleton instance
    Configuration._instance = None
    yield
    Configuration._instance = None


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


class TestConfigurationIntegration:
    """Test integration of all config components."""

    def test_configuration_singleton(self):
        """Test that cfg is a singleton instance."""
        # cfg is the Configuration class itself due to metaclass
        assert cfg is Configuration
        # The actual instance should be accessible
        instance = cfg._get_instance()
        assert isinstance(instance, Configuration)

    def test_configuration_hot_swappable_modules(self):
        """Test that all hot-swappable modules are properly configured."""
        assert isinstance(cfg.representation, RepresentationConfig)
        assert isinstance(cfg.dynamics, DynamicsConfig)
        assert isinstance(cfg.prediction, PredictionConfig)
        assert isinstance(cfg.memory, MemoryConfig)

    def test_configuration_validation(self):
        """Test that configuration validation works."""
        # This should not raise any errors
        cfg._validate()


class TestConfigurationLoading:
    """Test configuration loading and saving functionality."""

    def test_load_default_config(self):
        """Test loading default configuration."""
        # Reset singleton
        Configuration._instance = None

        # cfg should be the Configuration class
        assert cfg is Configuration
        # The instance should be created when accessed
        instance = cfg._get_instance()
        assert isinstance(instance, Configuration)

    def test_load_from_json(self):
        """Test loading configuration from JSON file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config_data = {"CuMind": {"networks": {"hidden_state_dim": 256}, "env": {"name": "TestEnv", "action_space_size": 4, "observation_shape": [8]}, "seed": 123}}
            import json

            json.dump(config_data, f)
            config_path = f.name

        try:
            # Load the config
            cfg.load(config_path)

            # Verify the loaded values
            assert cfg.networks.hidden_state_dim == 256
            assert cfg.env.name == "TestEnv"
            assert cfg.env.action_space_size == 4
            assert cfg.env.observation_shape == (8,)
            assert cfg.seed == 123
        finally:
            os.unlink(config_path)

    def test_save_config(self):
        """Test saving configuration to JSON file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config_path = f.name

        try:
            # Save the current config
            cfg.save(config_path)

            # Verify file was created and contains valid JSON
            assert os.path.exists(config_path)
            with open(config_path, "r") as f:
                import json

                saved_config = json.load(f)
                assert "CuMind" in saved_config
        finally:
            os.unlink(config_path)


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
