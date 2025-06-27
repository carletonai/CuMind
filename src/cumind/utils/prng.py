import threading
from typing import Optional, Tuple, Union

import jax

from .logger import log


class PRNGManager:
    """
    Singleton PRNG manager for JAX using new-style keys.
    Tracks an internal key chain for reproducibility,
    never exposing the current key directly to users.
    """

    _instance: Optional["PRNGManager"] = None
    _lock: threading.RLock = threading.RLock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                obj = super().__new__(cls)
                obj._key = None
                obj._seed = None
                obj._impl = None
                cls._instance = obj
                log.debug("PRNGManager singleton instance created")
        return cls._instance

    @classmethod
    def instance(cls):
        """Get the singleton instance of PRNGManager."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def _init(self, seed: int, impl: Union[str, None] = None):
        """
        Initialize the PRNG key with a seed and (optionally) a PRNG implementation.

        Args:
            seed: Integer seed for PRNG initialization
            impl: Optional PRNG implementation string (e.g., 'threefry2x32')
        """
        with type(self)._lock:
            if impl is not None:
                self._key = jax.random.key(seed, impl=impl)
                log.info(f"PRNG initialized with seed: {seed}, implementation: {impl}")
            else:
                self._key = jax.random.key(seed)
                log.info(f"PRNG initialized with seed: {seed} (default implementation)")
            self._seed = seed
            self._impl = impl

    def _split(self, num: Union[int, Tuple[int, ...]]):
        """
        Split the internal key, update the chain, and return new subkey(s).
        The internal key is updated, so every call advances the sequence.

        Args:
            num: Number of keys to generate (int or tuple of ints).

        Returns:
            - If int: returns a JAX key array of shape (num, ...)
            - If tuple: returns a JAX key array of shape (*num, ...)

        Raises:
            RuntimeError: If PRNG is not initialized
            ValueError: If num is invalid
        """
        with type(self)._lock:
            if self._key is None or self._seed is None:
                log.error("Attempted to split PRNG key before initialization")
                raise RuntimeError("PRNG not initialized. Call key(seed) first.")

            if isinstance(num, int):
                if num <= 0:
                    raise ValueError("num must be a positive integer")
                split_shape = (num + 1,)
                log.debug(f"Generating {num} PRNG subkey(s) (int split)")
            elif isinstance(num, tuple):
                if not num or not all(isinstance(x, int) and x > 0 for x in num):
                    raise ValueError("tuple must be non-empty and all elements positive ints")
                split_shape = (num[0] + 1,) + num[1:]
                log.debug(f"Generating PRNG subkey(s) with shape {num} (tuple split)")
            else:
                raise ValueError("num must be int or tuple of ints")

            keys = jax.random.split(self._key, split_shape)
            self._key = keys[0]
            return keys[1:]

    def __call__(self, x: Union[int, Tuple[int, ...]] = 1, impl: Union[str, None] = None) -> Optional[jax.Array]:
        """
        Main interface for PRNG operations.

        Args:
            x: If int and PRNG not initialized, used as seed. Otherwise, number of keys to generate.
            impl: Optional PRNG implementation (only used during initialization)

        Returns:
            Generated PRNG subkey(s) or None if initializing

        Raises:
            RuntimeError: If PRNG is not initialized and x is not a valid seed
        """
        with self._lock:
            # init case
            if self._key is None and self._seed is None and isinstance(x, int):
                log.info(f"Initializing PRNG with seed: {x}")
                self._init(x, impl)
                return None
            # incorrect invariant case
            if self._key is None or self._seed is None:
                log.error("Attempted to split PRNG key before initialization")
                raise RuntimeError("PRNG not initialized. Call key(seed) first.")
            # split case
            return self._split(x)


# User-facing singleton instance
key = PRNGManager.instance()

# Example usage:
# from cumind.utils.prng import key
# key(config.seed)      # initialize with seed
# subkey = key()        # returns 1 key
# subkeys = key(5)      # returns 5 keys
# kgrid = key((2, 3))   # returns (2, 3, 2) keys
