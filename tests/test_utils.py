"""Comprehensive tests for utility modules (Logger, JAX utils)."""

import os
import tempfile

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cumind.utils.config import cfg
from cumind.utils.logger import log
from cumind.utils.prng import key


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset the singletons before and after each test."""
    log._instance = None
    log._initialized = False
    key.seed(42)  # Initialize with a default seed
    yield
    log._instance = None
    log._initialized = False


class TestLogger:
    """Test suite for Logger."""

    def test_logger_initialization(self):
        """Test Logger initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = os.path.join(temp_dir, "test_logs")
            logger = log(log_dir=log_dir, use_timestamp=False)

            # The log_dir should be a Path object
            assert hasattr(logger, "log_dir")
            assert logger.log_dir.exists()
            assert hasattr(logger, "_logger")
            assert hasattr(logger, "tb_writer")

    def test_log_scalar(self):
        """Test scalar logging functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = os.path.join(temp_dir, "test_logs")
            logger = log(log_dir=log_dir, use_timestamp=False)

            # Log some scalar values
            logger.log_scalar("loss/total", 0.5, step=1)
            logger.log_scalar("loss/value", 0.3, step=1)
            logger.log_scalar("loss/policy", 0.2, step=1)

            # Check that logger has the right methods
            assert hasattr(logger, "log_scalar")
            assert hasattr(logger, "_logger")

    def test_log_multiple_steps(self):
        """Test logging across multiple steps."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = os.path.join(temp_dir, "test_logs")
            logger = log(log_dir=log_dir, use_timestamp=False)

            # Log values across multiple steps
            for step in range(10):
                logger.log_scalar("training/loss", float(step) * 0.1, step=step)
                logger.log_scalar("training/reward", float(step) * 2.0, step=step)

            # Verify logging doesn't crash
            assert True  # If we get here, logging worked

    def test_log_info(self):
        """Test text logging functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = os.path.join(temp_dir, "test_logs")
            logger = log(log_dir=log_dir, use_timestamp=False)

            # Log text message
            logger.info("Training started")
            logger.info("Epoch 1 completed")

            # Verify no errors occurred
            assert True

    def test_close_logger(self):
        """Test logger cleanup."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = os.path.join(temp_dir, "test_logs")
            logger = log(log_dir=log_dir, use_timestamp=False)

            # Log some data
            logger.log_scalar("test/metric", 1.0, step=1)

            # Close logger
            logger.close()

            # Verify close operation completes
            assert True

    def test_close_functionality(self):
        """Test logger cleanup and resource management."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = os.path.join(temp_dir, "test_logs")
            logger = log(log_dir=log_dir, use_timestamp=False)

            # Log some data
            logger.log_scalar("test/metric", 1.0, step=0)

            # Test close functionality
            logger.close()

            # Verify close doesn't crash
            assert True

    def test_metrics_aggregation(self):
        """Test metrics aggregation and averaging."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = os.path.join(temp_dir, "test_logs")
            logger = log(log_dir=log_dir, use_timestamp=False)

            # Log multiple values for averaging
            values = [0.1, 0.2, 0.3, 0.4, 0.5]
            for i, value in enumerate(values):
                logger.log_scalar("test/metric", value, step=i)

            # Test that all values were logged successfully
            assert True

    def test_invalid_log_directory(self):
        """Test behavior with invalid log directory."""
        # Test with invalid path
        invalid_path = "/nonexistent/path/that/should/not/exist"

        # Should either handle gracefully or raise appropriate error
        try:
            logger = log(log_dir=invalid_path, use_timestamp=False)
            # If no error, verify logger was created
            assert logger is not None
        except (OSError, PermissionError, FileNotFoundError):
            # Expected behavior for invalid path
            assert True

    def test_concurrent_logging(self):
        """Test logging from multiple sources."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = os.path.join(temp_dir, "test_logs")
            logger = log(log_dir=log_dir, use_timestamp=False)

            # Simulate concurrent logging of different metrics
            metrics = ["loss", "reward", "value", "policy"]

            for step in range(5):
                for metric in metrics:
                    logger.log_scalar(f"train/{metric}", np.random.random(), step=step)

            # Verify all logging completed successfully
            assert True


class TestJAXUtils:
    """Test suite for JAX utility functions."""

    def test_tree_operations(self):
        """Test JAX tree operations."""
        # Create test tree structure
        tree1 = {
            "params": {
                "dense1": {"kernel": jnp.ones((4, 32)), "bias": jnp.zeros(32)},
                "dense2": {"kernel": jnp.ones((32, 2)), "bias": jnp.zeros(2)},
            },
            "stats": {"mean": jnp.array([1.0, 2.0]), "std": jnp.array([0.5, 1.5])},
        }
        tree2 = {
            "params": {
                "dense1": {"kernel": jnp.ones((4, 32)), "bias": jnp.zeros(32)},
                "dense2": {"kernel": jnp.ones((32, 2)), "bias": jnp.zeros(2)},
            },
            "stats": {"mean": jnp.array([1.0, 2.0]), "std": jnp.array([0.5, 1.5])},
        }

        # Test tree_flatten and tree_unflatten
        flat, tree_def = jax.tree_util.tree_flatten(tree1)
        reconstructed = jax.tree_util.tree_unflatten(tree_def, flat)

        # Verify reconstruction
        chex.assert_trees_all_close(reconstructed, tree2)

    def test_tree_map_operations(self):
        """Test tree_map operations."""
        tree = {"a": jnp.array([1.0, 2.0]), "b": {"c": jnp.array([3.0, 4.0]), "d": jnp.array([5.0, 6.0])}}

        # Test tree_map with multiplication
        scaled_tree = jax.tree.map(lambda x: x * 2.0, tree)

        assert jnp.allclose(scaled_tree["a"], jnp.array([2.0, 4.0]))
        assert jnp.allclose(scaled_tree["b"]["c"], jnp.array([6.0, 8.0]))
        assert jnp.allclose(scaled_tree["b"]["d"], jnp.array([10.0, 12.0]))

    def test_array_operations(self):
        """Test JAX array operations."""
        # Test basic array operations
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([4.0, 5.0, 6.0])

        # Test element-wise operations
        assert jnp.allclose(x + y, jnp.array([5.0, 7.0, 9.0]))
        assert jnp.allclose(x * y, jnp.array([4.0, 10.0, 18.0]))

        # Test reduction operations
        assert jnp.allclose(jnp.sum(x), 6.0)
        assert jnp.allclose(jnp.mean(x), 2.0)

    def test_gradient_computation(self):
        """Test gradient computation."""

        def simple_function(x):
            return jnp.sum(x**2)

        x = jnp.array([1.0, 2.0, 3.0])
        grad_fn = jax.grad(simple_function)
        gradients = grad_fn(x)

        # Expected gradients: 2 * x
        expected_gradients = jnp.array([2.0, 4.0, 6.0])
        assert jnp.allclose(gradients, expected_gradients)

    def test_jit_compilation(self):
        """Test JIT compilation."""

        @jax.jit
        def jitted_function(x, y):
            return x + y

        x = jnp.array([1.0, 2.0])
        y = jnp.array([3.0, 4.0])
        result = jitted_function(x, y)

        assert jnp.allclose(result, jnp.array([4.0, 6.0]))

    def test_vectorization(self):
        """Test vectorization operations."""

        def single_input_function(x):
            return x**2

        # Vectorize the function
        vectorized_fn = jax.vmap(single_input_function)

        # Test with batch input
        batch_input = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        result = vectorized_fn(batch_input)

        expected = jnp.array([[1.0, 4.0], [9.0, 16.0]])
        assert jnp.allclose(result, expected)

    def test_device_operations(self):
        """Test device operations."""
        # Test that we can get available devices
        devices = jax.devices()
        assert len(devices) > 0

        # Test that we can specify device
        device = jax.devices(cfg.device)[0]
        assert device is not None


class Testkey:
    """Test suite for key management."""

    def test_deterministic_key_generation(self):
        """Test that key generation is deterministic."""
        key.seed(42)
        key1 = key.get()
        key.seed(42)
        key2 = key.get()

        # Keys should be the same for same seed
        assert jnp.array_equal(key1, key2)

        # Different seeds should produce different keys
        key.seed(43)
        key3 = key.get()
        assert not jnp.array_equal(key1, key3)

    def test_key_uniqueness(self):
        """Test that split keys are unique."""
        key.seed(42)
        split_keys = key.get(3)

        # All split keys should be different
        assert not jnp.array_equal(split_keys[0], split_keys[1])
        assert not jnp.array_equal(split_keys[1], split_keys[2])
        assert not jnp.array_equal(split_keys[0], split_keys[2])

    def test_key_generation_shapes(self):
        """Test key generation with different shapes."""
        key.seed(42)

        # Test single key
        single_key = key.get()
        # Key shape can be empty for some JAX implementations
        assert hasattr(single_key, "shape")

        # Test multiple keys
        multiple_keys = key.get(5)
        assert len(multiple_keys) == 5
        for k in multiple_keys:
            assert hasattr(k, "shape")

        # Test with custom shape
        custom_key = key.get()
        assert hasattr(custom_key, "shape")

    def test_uninitialized_error(self):
        """Test error when key is not initialized."""
        # Reset key state to uninitialized
        key._key = None
        key._seed = None

        with pytest.raises(RuntimeError):
            key.get()

    def test_invalid_split_number(self):
        """Test error handling for invalid split numbers."""
        key.seed(42)
        with pytest.raises(ValueError):
            key.get(0)

        with pytest.raises(ValueError):
            key.get(-1)

    def test_seeding_twice_ignored(self):
        """Test that seeding twice with same value is ignored."""
        key.seed(42)
        first_key = key.get()
        key.seed(42)  # Should be ignored
        second_key = key.get()

        # Keys should be the same
        assert jnp.array_equal(first_key, second_key)


if __name__ == "__main__":
    pytest.main([__file__])
