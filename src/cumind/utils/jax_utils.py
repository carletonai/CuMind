"""JAX utility functions for CuMind."""

from typing import Any, Callable, Sequence, Tuple, cast

import chex
import jax
import jax.numpy as jnp

from cumind.utils.logger import log


def tree_stack(trees: Sequence[Any]) -> Any:
    """Stack a sequence of PyTrees along a new first axis.

    Args:
        trees: Sequence of PyTrees with the same structure

    Returns:
        PyTree with arrays stacked along new axis 0
    """
    log.debug(f"Stacking {len(trees)} trees.")
    return jax.tree_util.tree_map(lambda *arrays: jnp.stack(arrays, axis=0), *trees)


def tree_unstack(tree: Any) -> Tuple[Any, ...]:
    """Unstack a PyTree along the first axis.

    Args:
        tree: PyTree with arrays to unstack

    Returns:
        Tuple of PyTrees unstacked along axis 0
    """
    num_leaves = len(jax.tree_util.tree_leaves(tree))
    log.debug(f"Unstacking tree with {num_leaves} leaves.")
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    unstacked_leaves = [jnp.split(leaf, leaf.shape[0], axis=0) for leaf in leaves]
    unstacked_trees = []
    for i in range(leaves[0].shape[0]):
        tree_leaves = [leaf_splits[i].squeeze(0) for leaf_splits in unstacked_leaves]
        unstacked_trees.append(jax.tree_util.tree_unflatten(treedef, tree_leaves))
    return tuple(unstacked_trees)


def batched_apply[T](fn: Callable[[Any], T], *args: Any, batch_size: int) -> T:
    """Apply function in batches to avoid memory issues.

    Args:
        fn: Function to apply
        *args: Arguments to batch and pass to fn
        batch_size: Size of each batch

    Returns:
        Concatenated results from all batches
    """
    total_size = jax.tree_util.tree_leaves(args[0])[0].shape[0]

    if total_size <= batch_size:
        log.debug("Total size is smaller than batch size, applying function directly.")
        return fn(*args)

    log.info(f"Applying function in batches of size {batch_size} for total size {total_size}.")
    results = []
    for i in range(0, total_size, batch_size):
        log.debug(f"Processing batch {i // batch_size + 1}/{total_size // batch_size + 1}")
        end_idx = min(i + batch_size, total_size)
        batch_args = jax.tree_util.tree_map(lambda x: x[i:end_idx], args)
        batch_result = fn(*batch_args)
        results.append(batch_result)

    return cast(T, jax.tree_util.tree_map(lambda *arrays: jnp.concatenate(arrays, axis=0), *results))


def safe_normalize(x: chex.Array, axis: int = -1, epsilon: float = 1e-8) -> chex.Array:
    """Safely normalize array to avoid division by zero.

    Args:
        x: Array to normalize
        axis: Axis along which to normalize
        epsilon: Small value to avoid division by zero

    Returns:
        Normalized array
    """
    norm = jnp.linalg.norm(x, axis=axis, keepdims=True)
    return x / jnp.maximum(norm, epsilon)


def vectorized_n_step_return(rewards: jnp.ndarray, values: jnp.ndarray, n_steps: int, discount: float) -> jnp.ndarray:
    """Compute the n-step return for a sequence of rewards and values.

    This function calculates the n-step return using a vectorized approach,
    replacing the need for a Python loop. It is designed to work with JAX for
    efficient computation on accelerators.

    Args:
        rewards: Array of rewards
        values: Array of value estimates
        n_steps: Number of steps to consider for the n-step return
        discount: Discount factor for future rewards

    Returns:
        Array containing the n-step returns
    """
    idx = jnp.arange(n_steps)
    mask = idx < rewards.shape[0]
    rewards = jnp.where(mask, rewards[:n_steps], 0)
    discounts = discount**idx
    n_step_return = jnp.sum(rewards * discounts)
    if rewards.shape[0] > n_steps:
        n_step_return += (discount**n_steps) * values[n_steps]
    return n_step_return


def vmap_n_step_return(rewards: jnp.ndarray, values: jnp.ndarray, n_steps: int, discount: float) -> chex.Array:
    """Computes n-step returns for batches of rewards and values using JAX vmap."""
    return cast(chex.Array, jax.vmap(vectorized_n_step_return, in_axes=(0, 0, None, None))(rewards, values, n_steps, discount))


def pmap_n_step_return(rewards: jnp.ndarray, values: jnp.ndarray, n_steps: int, discount: float) -> chex.Array:
    """Computes n-step returns for batches of rewards and values using JAX vmap."""
    return cast(chex.Array, jax.pmap(vectorized_n_step_return, in_axes=(0, 0, None, None))(rewards, values, n_steps, discount))
