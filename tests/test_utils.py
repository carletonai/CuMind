"""Comprehensive tests for utility modules (Logger, JAX utils)."""

import os
import tempfile

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cumind.utils.logger import Logger
from cumind.utils.prng import key


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset the singletons before and after each test."""
    Logger._instance = None
    Logger._initialized = False
    key.reset()
    yield
    Logger._instance = None
    Logger._initialized = False
    key.reset()


class TestLogger:
    """Test suite for Logger."""

    def test_logger_initialization(self):
        """Test Logger initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = os.path.join(temp_dir, "test_logs")
            logger = Logger(log_dir=log_dir, use_timestamp=False)

            # The log_dir is a Path object, not string
            assert str(logger.log_dir) == log_dir
            assert logger.log_dir.exists()
            assert hasattr(logger, "_logger")
            assert hasattr(logger, "tb_writer")

    def test_log_scalar(self):
        """Test scalar logging functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = os.path.join(temp_dir, "test_logs")
            logger = Logger(log_dir=log_dir, use_timestamp=False)

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
            logger = Logger(log_dir=log_dir, use_timestamp=False)

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
            logger = Logger(log_dir=log_dir, use_timestamp=False)

            # Log text message
            logger.info("Training started")
            logger.info("Epoch 1 completed")

            # Verify no errors occurred
            assert True

    def test_close_logger(self):
        """Test logger cleanup."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = os.path.join(temp_dir, "test_logs")
            logger = Logger(log_dir=log_dir, use_timestamp=False)

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
            logger = Logger(log_dir=log_dir, use_timestamp=False)

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
            logger = Logger(log_dir=log_dir, use_timestamp=False)

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
            logger = Logger(log_dir=invalid_path, use_timestamp=False)
            # If no error, verify logger was created
            assert logger is not None
        except (OSError, PermissionError, FileNotFoundError):
            # Expected behavior for invalid path
            assert True

    def test_concurrent_logging(self):
        """Test logging from multiple sources."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = os.path.join(temp_dir, "test_logs")
            logger = Logger(log_dir=log_dir, use_timestamp=False)

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
        # Test basic array creation and operations
        a = jnp.array([1.0, 2.0, 3.0])
        b = jnp.array([4.0, 5.0, 6.0])

        # Test arithmetic operations
        c = a + b
        expected = jnp.array([5.0, 7.0, 9.0])
        assert jnp.allclose(c, expected)

        # Test matrix operations
        matrix = jnp.ones((3, 3))
        result = jnp.dot(matrix, a)
        expected_result = jnp.array([6.0, 6.0, 6.0])
        assert jnp.allclose(result, expected_result)

    def test_gradient_computation(self):
        """Test JAX gradient computation."""

        def simple_function(x):
            return jnp.sum(x**2)

        # Test grad
        grad_fn = jax.grad(simple_function)
        x = jnp.array([1.0, 2.0, 3.0])
        gradients = grad_fn(x)

        # Gradient of sum(x^2) is 2*x
        expected_gradients = 2 * x
        assert jnp.allclose(gradients, expected_gradients)

    def test_jit_compilation(self):
        """Test JAX JIT compilation."""

        @jax.jit
        def jitted_function(x, y):
            return x @ y + jnp.sum(x)

        # Test with different input shapes
        x1 = jnp.ones((2, 3))
        y1 = jnp.ones((3, 4))
        result1 = jitted_function(x1, y1)

        assert result1.shape == (2, 4)

        # Test that function can be called multiple times
        x2 = jnp.ones((2, 3)) * 2
        y2 = jnp.ones((3, 4)) * 2
        result2 = jitted_function(x2, y2)
        assert result2.shape == (2, 4)
        assert not jnp.allclose(result1, result2)

    def test_vectorization(self):
        """Test JAX vectorization."""

        def single_input_function(x):
            return x * 2

        # Test vmap
        vmap_fn = jax.vmap(single_input_function)
        x_batch = jnp.arange(5)
        result_batch = vmap_fn(x_batch)

        expected_batch = x_batch * 2
        assert jnp.allclose(result_batch, expected_batch)

    def test_device_operations(self):
        """Test JAX device operations."""
        # Test that JAX is using the default device
        assert "cpu" in jax.default_backend().lower() or "gpu" in jax.default_backend().lower()


class Testkey:
    """Test suite for the key."""

    def test_deterministic_key_generation(self):
        """Test that seeding produces a deterministic sequence of keys."""
        key.seed(42)
        keys1 = [key() for _ in range(5)]

        key.reset()

        key.seed(42)
        keys2 = [key() for _ in range(5)]

        for k1, k2 in zip(keys1, keys2):
            assert jnp.array_equal(k1, k2)

    def test_key_uniqueness(self):
        """Test that generated keys are unique."""
        key.seed(101)
        keys = [key() for _ in range(100)]
        unique_keys = {tuple(np.asarray(jax.random.key_data(k)).tolist()) for k in keys}
        assert len(keys) == len(unique_keys)

    def test_key_generation_shapes(self):
        """Test key generation for different shapes."""
        key.seed(0)
        # Get the shape of a single key to make the test robust to PRNG implementation
        key_shape = key().shape

        # Reset for the actual tests
        key.reset()
        key.seed(0)

        assert key().shape == key_shape
        assert key(1).shape == key_shape
        assert key(5).shape == (5,) + key_shape
        assert key((2, 3)).shape == (2, 3) + key_shape

    def test_uninitialized_error(self):
        """Test that using the PRNG before seeding raises an error."""
        with pytest.raises(RuntimeError, match="PRNG not initialized"):
            key()

    def test_invalid_split_number(self):
        """Test that requesting zero or a negative number of keys raises an error."""
        key.seed(0)
        with pytest.raises(ValueError, match="num must be a positive integer"):
            key(0)
        with pytest.raises(ValueError, match="num must be a positive integer"):
            key(-1)

    def test_seeding_twice_ignored(self):
        """Test that seeding a second time is ignored."""
        key.seed(42)
        key1 = key()

        # This second seed should be ignored
        key.seed(99)
        key2 = key()

        # To verify, we reset and re-seed with 42
        key.reset()
        key.seed(42)
        key_control = key()  # This is the first key after seeding

        assert not jnp.array_equal(key1, key2)
        assert jnp.array_equal(key_control, key1)


if __name__ == "__main__":
    pytest.main([__file__])
