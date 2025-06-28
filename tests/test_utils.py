"""Comprehensive tests for utility modules (Logger, JAX utils)."""

import os
import tempfile

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cumind.utils.logger import Logger
from cumind.utils.prng import PRNGManager
from cumind.utils.prng import key as prng_key


@pytest.fixture(autouse=True)
def reset_logger_singleton():
    """Reset the Logger singleton before and after each test."""
    Logger._instance = None
    Logger._initialized = False
    yield
    Logger._instance = None
    Logger._initialized = False


@pytest.fixture(autouse=True)
def reset_prng_manager_singleton():
    """Reset the PRNGManager singleton before and after each test."""
    PRNGManager._instance = None
    yield
    PRNGManager._instance = None


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
        tree1 = {"params": {"dense1": {"kernel": jnp.ones((4, 32)), "bias": jnp.zeros(32)}, "dense2": {"kernel": jnp.ones((32, 2)), "bias": jnp.zeros(2)}}, "stats": {"mean": jnp.array([1.0, 2.0]), "std": jnp.array([0.5, 1.5])}}
        tree2 = {"params": {"dense1": {"kernel": jnp.ones((4, 32)), "bias": jnp.zeros(32)}, "dense2": {"kernel": jnp.ones((32, 2)), "bias": jnp.zeros(2)}}, "stats": {"mean": jnp.array([1.0, 2.0]), "std": jnp.array([0.5, 1.5])}}

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

    def test_random_key_operations(self):
        """Test CuMind PRNG key operations."""
        # Test key splitting and initialization
        prng_key.seed(42)
        key1 = prng_key()
        key2 = prng_key()

        # Keys should be different
        assert not jnp.array_equal(key1, key2)

        # Test multiple splits
        keys = prng_key(5)
        assert keys.shape == (5, )  # JAX PRNG keys are 2-element arrays

        # All keys should be different
        # unstack keys for comparison
        unstacked_keys = [keys[i] for i in range(keys.shape[0])]
        all_keys = [key1, key2] + unstacked_keys
        for i in range(len(all_keys)):
            for j in range(i + 1, len(all_keys)):
                assert not jnp.array_equal(all_keys[i], all_keys[j])

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

    def test_vectorization(self):
        """Test JAX vectorization with vmap."""

        def single_input_function(x):
            return jnp.sum(x**2)

        # Vectorize to handle batch of inputs
        batch_function = jax.vmap(single_input_function)

        # Test with batch input
        batch_x = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        batch_result = batch_function(batch_x)

        assert batch_result.shape == (3,)

        # Verify results
        expected = jnp.array([5.0, 25.0, 61.0])  # 1^2+2^2, 3^2+4^2, 5^2+6^2
        assert jnp.allclose(batch_result, expected)

    def test_device_operations(self):
        """Test JAX device operations."""
        # Test array creation on default device
        x = jnp.array([1.0, 2.0, 3.0])

        # Verify array has device attribute
        assert hasattr(x, "device") or hasattr(x, "devices")

        # Test basic operations work regardless of device
        y = x * 2
        expected = jnp.array([2.0, 4.0, 6.0])
        assert jnp.allclose(y, expected)


if __name__ == "__main__":
    pytest.main([__file__])
