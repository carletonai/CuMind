"""PRNG utilities for JAX key management."""

import threading
from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp

from .logger import log


class PRNGManager:
    """Singleton PRNG manager for JAX using new-style keys.

    Usage:
        from cumind.utils.prng import key
        key.seed(config.seed)      # initialize with seed
        subkey = key()             # returns 1 key
        subkeys = key(5)           # returns 5 keys
        kgrid = key((2, 3))        # returns (2, 3, 2) keys
    """

    _instance: Optional["PRNGManager"] = None
    _lock = threading.RLock()
    _initialized: bool = False

    def __new__(cls) -> "PRNGManager":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize PRNGManager state."""
        if self._initialized:
            return

        self._reset()
        self._initialized = True
        log.debug("PRNGManager singleton instance created")

    def _reset(self) -> None:
        """Reset the internal state of the PRNG manager."""
        self._key: Optional[jax.Array] = None
        self._seed: Optional[int] = None
        self._impl: Optional[str] = None

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance._reset()

    @classmethod
    def instance(cls) -> "PRNGManager":
        """Get singleton instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def seed(self, seed: int, impl: Union[str, None] = None) -> None:
        """Initialize PRNG key with seed.

        Args:
            seed: a 64- or 32-bit integer used as the value of the key
            impl: optional string specifying the PRNG implementation (e.g. 'threefry2x32')
        """
        with type(self)._lock:
            if self._key is not None or self._seed is not None:
                log.warning("PRNG manager already seeded. Ignoring new seed.")
                return

            if impl is not None:
                self._key = jax.random.key(seed, impl=impl)
                log.info(f"PRNG initialized with seed: {seed}, implementation: {impl}")
            else:
                self._key = jax.random.key(seed)
                log.info(f"PRNG initialized with seed: {seed}")
            self._seed = seed
            self._impl = impl

    def _split(self, num: Union[int, Tuple[int, ...]]) -> jax.Array:
        """
        Split internal key and return subkey(s).

        Args:
            num: Number of keys to generate (int or tuple of ints)

        Returns:
            JAX key array of requested shape

        Raises:
            RuntimeError: If PRNG is not initialized
            ValueError: If num is invalid
        """
        with type(self)._lock:
            if self._key is None or self._seed is None:
                log.error("Attempted to split PRNG key before initialization")
                raise RuntimeError("PRNG not initialized. Call key.seed(value) first.")

            log.debug(f"Starting PRNG key split operation. Input num: {num}, Current key shape: {self._key.shape}")

            if isinstance(num, int):
                if num <= 0:
                    raise ValueError("num must be a positive integer")
                shape: Tuple[int, ...] = (num,)
                log.debug(f"Processing int input. Generated shape: {shape}")
            elif isinstance(num, tuple):
                if not all(isinstance(x, int) and x > 0 for x in num):
                    raise ValueError("tuple must be non-empty and all elements positive ints")
                shape = num
                log.debug(f"Processing tuple input. Generated shape: {shape}")
            else:
                raise ValueError("num must be int or tuple of ints")

            prod = int(jnp.prod(jnp.array(shape)))
            log.debug(f"Calculated total number of keys needed: {prod}")

            log.debug(f"Before split - Key shape: {self._key.shape}")
            self._key, *subkeys = jax.random.split(self._key, prod + 1)
            log.debug(f"After split - Main key shape: {self._key.shape}, Number of subkeys: {len(subkeys)}")

            if prod == 1:
                result = subkeys[0]
                log.debug(f"Single key case - Result shape: {result.shape}")
            else:
                if self._key is None:
                    raise RuntimeError("PRNG key should be initialized before splitting")
                result = jnp.array(subkeys).reshape(shape + self._key.shape)
                log.debug(f"Multiple keys case - Reshaped result shape: {result.shape}")

            log.debug(f"Final result shape: {result.shape}, Returning as JAX array")
            return jnp.asarray(result)

    def __call__(self, x: Union[int, Tuple[int, ...]] = 1) -> jax.Array:
        """
        Main interface for PRNG key generation.

        Args:
            x: Number of keys to generate. If an int, returns `x` keys.
               If a tuple, generates keys of that shape. Defaults to 1.

        Returns:
            A generated PRNG subkey (or keys).

        Raises:
            RuntimeError: If trying to generate keys before initialization.
        """
        with self._lock:
            if self._key is None or self._seed is None:
                log.error("Attempted to split PRNG key before initialization")
                raise RuntimeError("PRNG not initialized. Call key.seed(value) first.")
            return self._split(x)


# Singleton instance
key = PRNGManager.instance()
