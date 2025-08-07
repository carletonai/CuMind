"""PRNG utilities for JAX key management."""

import sys
import threading
from typing import Optional, Sequence, Tuple, Union, cast

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike, DTypeLike

from cumind.utils.logger import log


class KeyManager:
    """Singleton PRNG manager for JAX using new-style keys.

    Usage:
        from cumind.utils.prng import key
        key.seed(42)           # initialize with seed
        subkey = key.get()     # returns 1 key
        subkeys = key.get(5)   # returns 5 keys
        kgrid = key.get((2,3)) # returns (2,3,2) keys
    """

    _instance: Optional["KeyManager"] = None
    _key: Optional[ArrayLike] = None
    _seed: Optional[int] = None
    _impl: Optional[str] = None
    _lock = threading.RLock()

    def __new__(cls) -> "KeyManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._key = None
                    cls._seed = None
                    cls._impl = None
                    log.debug("KeyManager singleton instance created")
        return cls._instance

    @classmethod
    def seed(cls, seed: int, impl: Optional[str] = None) -> None:
        """Initialize PRNG key with seed."""
        with cls._lock:
            cls._seed = seed
            cls._impl = impl
            if impl is not None:
                cls._key = jax.random.key(seed, impl=impl)
                log.info(f"PRNG initialized with seed: {seed}, implementation: {impl}")
            else:
                cls._key = jax.random.key(seed)
                log.info(f"PRNG initialized with seed: {seed}")

    @classmethod
    def get(cls, num: Union[int, Tuple[int, ...]] = 1) -> ArrayLike:
        """Get subkey(s) from the PRNG. num can be int or tuple of ints."""
        with cls._lock:
            if cls._key is None or cls._seed is None:
                log.error("Attempted to split PRNG key before initialization")
                raise RuntimeError("PRNG not initialized. Call key.seed(value) first.")

            if isinstance(num, int):
                if num <= 0:
                    raise ValueError("num must be a positive integer")
                shape: Tuple[int, ...] = (num,)
            elif isinstance(num, tuple):
                if not all(isinstance(x, int) and x > 0 for x in num):
                    raise ValueError("tuple must be non-empty and all elements positive ints")
                shape = num
            else:
                raise ValueError("num must be int or tuple of ints")

            prod = int(jnp.prod(jnp.array(shape)))
            split_keys = jax.random.split(cls._key, prod + 1)
            cls._key = split_keys[0]
            subkeys = split_keys[1:]

            if prod == 1:
                return subkeys[0]
            else:
                assert cls._key is not None
                return jnp.array(subkeys).reshape(shape + cls._key.shape)

    @classmethod
    def randint(cls, min: ArrayLike = int(), max: ArrayLike = sys.maxsize, shape: Sequence[int] = (1,), dtype: DTypeLike = int) -> int:
        """Return a random integer between minval and maxval."""
        with cls._lock:
            if cls._key is None or cls._seed is None:
                log.error("Attempted to use PRNG before initialization")
                raise RuntimeError("PRNG not initialized. Call key.seed(value) first.")
            return cast(int, jax.random.randint(cls._key, shape, min, max, dtype).item())


# Alias
key = KeyManager
