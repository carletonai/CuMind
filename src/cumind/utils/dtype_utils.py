"""Utilities for handling data types from configuration."""

from typing import Any, Union

import jax.numpy as jnp
import numpy as np


def get_dtype(dtype_str: str) -> Any:
    """Convert string dtype to JAX/NumPy dtype.

    Args:
        dtype_str: String representation of dtype (e.g., "float32", "int32")

    Returns:
        The corresponding JAX/NumPy dtype

    Raises:
        ValueError: If dtype_str is not recognized
    """
    dtype_map = {
        "float32": jnp.float32,
        "float16": jnp.float16,
        "bfloat16": jnp.bfloat16,
        "int32": jnp.int32,
        "int64": jnp.int64,
    }

    if dtype_str not in dtype_map:
        raise ValueError(f"Unknown dtype: {dtype_str}")

    return dtype_map[dtype_str]
