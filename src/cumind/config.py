"""Configuration for CuMind."""

from typing import Any, Dict, Optional, Tuple, Union

import chex
from flax import nnx


@chex.dataclass
class Config:
    """Hyperparameter configuration for the CuMind agent.

    This dataclass defines all configurable hyperparameters. For more details
    on tuning, see the project's documentation.
    """

    # Agent
    representation_network: Optional[Union[str, nnx.Module]] = "ResNet"
    dynamics_network: Optional[Union[str, nnx.Module]] = "MLPWithEmbedding"
    prediction_network: Optional[Union[str, nnx.Module]] = "MLPDual"

    # Network architecture
    hidden_state_dim: int = 128  # global

        # Representation network config
        representation_num_blocks: int = 2
        representation_conv_channels: int = 32

        # Dynamics network config
        dynamics_num_layers: int = 2

        # Prediction network config
        # (no additional parameters)

    # Training
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    target_update_frequency: int = 10000000
    checkpoint_interval: int = 100
    num_episodes: int = 500
    train_frequency: int = 5
    checkpoint_root_dir: str = "checkpoints"

    # MCTS
    num_simulations: int = 10
    c_puct: float = 1.0
    dirichlet_alpha: float = 0.3
    exploration_fraction: float = 0.25

    # Environment
    env_name: str = "CartPole-v1"
    action_space_size: int = 2
    observation_shape: Tuple[int, ...] = (4,)

    # Self-Play
    num_unroll_steps: int = 3
    td_steps: int = 5
    discount: float = 0.997

    # Memory
    memory_capacity: int = 1000
    min_memory_size: int = 3
    min_memory_pct: float = 0.00
    # Tree
    per_alpha: float = 0.6
    per_epsilon: float = 1e-6
    # Prioritized Buffer
    per_beta: float = 0.4

    # Data Types
    model_dtype: str = "float32"
    action_dtype: str = "int32"
    target_dtype: str = "float32"

    # Devices
    device_type: str = "cpu"

    # Other
    seed: int = 42
