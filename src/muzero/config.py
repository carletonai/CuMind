"""Configuration for MuZero."""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class MuZeroConfig:
    """Hyperparameter configuration for MuZero.

    Note:
        This dataclass lists all configurable hyperparameters for MuZero.
        For more details and optimization strategies, see:
        https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization
    """

    # Network architecture
    hidden_dim: int = 64
    num_blocks: int = 4

    # Training
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    # MCTS
    num_simulations: int = 50
    c_puct: float = 1.0
    dirichlet_alpha: float = 0.3
    exploration_fraction: float = 0.25

    # Environment
    action_space_size: int = 4
    observation_shape: Tuple[int, ...] = (84, 84, 3)

    # Training schedule
    num_unroll_steps: int = 5
    td_steps: int = 10
    discount: float = 0.997

    # Replay buffer
    replay_buffer_size: int = 10000
    min_replay_size: int = 1000
