"""Neural network components for MuZero."""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as functional


class ResidualBlock(nn.Module):
    """Residual block for the neural network."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = functional.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return functional.relu(x + residual)


class BaseEncoder(nn.Module):
    """Base class for observation encoders."""

    def __init__(self, observation_shape: Tuple[int, ...], hidden_dim: int, num_blocks: int):
        super().__init__()
        self.observation_shape = observation_shape
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class VectorEncoder(BaseEncoder):
    """Encoder for 1D vector observations (e.g., CartPole)."""

    def __init__(self, observation_shape: Tuple[int, ...], hidden_dim: int, num_blocks: int):
        super().__init__(observation_shape, hidden_dim, num_blocks)
        input_dim = observation_shape[0]

        # Build layers based on num_blocks
        layers = []
        current_dim = input_dim

        for i in range(num_blocks):
            layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.ReLU(),
                ]
            )
            current_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        output: torch.Tensor = self.encoder(observation)
        return output


class ConvEncoder(BaseEncoder):
    """Encoder for 3D image observations (e.g., Atari)."""

    def __init__(self, observation_shape: Tuple[int, ...], hidden_dim: int, num_blocks: int):
        super().__init__(observation_shape, hidden_dim, num_blocks)
        channels, height, width = observation_shape

        self.conv = nn.Conv2d(channels, hidden_dim, 3, padding=1)
        self.bn = nn.BatchNorm2d(hidden_dim)
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(num_blocks)])

        # Calculate output size after conv layers and add projection layer
        self.flatten_size = hidden_dim * height * width
        self.projection = nn.Linear(self.flatten_size, hidden_dim)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        x = functional.relu(self.bn(self.conv(observation)))
        for block in self.blocks:
            x = block(x)
        x = x.flatten(1)  # Flatten spatial dimensions
        return functional.relu(self.projection(x))  # Project to hidden_dim


class RepresentationNetwork(nn.Module):
    """Converts observations to hidden state representation."""

    def __init__(self, observation_shape: Tuple[int, ...], hidden_dim: int, num_blocks: int):
        super().__init__()
        self.encoder = self._create_encoder(observation_shape, hidden_dim, num_blocks)

    def _create_encoder(
        self, observation_shape: Tuple[int, ...], hidden_dim: int, num_blocks: int
    ) -> BaseEncoder:
        """Factory method to create the appropriate encoder based on observation shape."""
        if len(observation_shape) == 1:
            return VectorEncoder(observation_shape, hidden_dim, num_blocks)
        elif len(observation_shape) == 3:
            return ConvEncoder(observation_shape, hidden_dim, num_blocks)
        else:
            raise ValueError(
                f"Unsupported observation shape: {observation_shape}. "
                f"Expected 1D vector or 3D image (channels, height, width)."
            )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        output: torch.Tensor = self.encoder(observation)
        return output


class DynamicsNetwork(nn.Module):
    """Predicts next hidden state and reward given current state and action."""

    def __init__(self, hidden_dim: int, action_space_size: int, num_blocks: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Action embedding
        self.action_embedding = nn.Embedding(action_space_size, hidden_dim)

        # State transition
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                )
                for _ in range(num_blocks)
            ]
        )

        # Reward prediction
        self.reward_head = nn.Linear(hidden_dim, 1)

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Embed action and concatenate with state
        action_emb = self.action_embedding(action)
        x = torch.cat([state, action_emb], dim=-1)

        # Predict next state
        next_state = functional.relu(self.fc(x))
        for block in self.blocks:
            residual = next_state
            next_state = block(next_state) + residual

        # Predict reward
        reward = self.reward_head(next_state)

        return next_state, reward


class PredictionNetwork(nn.Module):
    """Predicts policy and value given hidden state."""

    def __init__(self, hidden_dim: int, action_space_size: int):
        super().__init__()
        self.policy_head = nn.Linear(hidden_dim, action_space_size)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        policy_logits = self.policy_head(state)
        value = self.value_head(state)
        return policy_logits, value


class MuZeroNetwork(nn.Module):
    """Complete MuZero neural network."""

    def __init__(
        self,
        observation_shape: Tuple[int, ...],
        action_space_size: int,
        hidden_dim: int = 64,
        num_blocks: int = 4,
    ):
        super().__init__()

        self.representation = RepresentationNetwork(observation_shape, hidden_dim, num_blocks)
        self.dynamics = DynamicsNetwork(hidden_dim, action_space_size, num_blocks)
        self.prediction = PredictionNetwork(hidden_dim, action_space_size)

    def initial_inference(
        self, observation: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Initial step: observation -> hidden state -> policy, value."""
        hidden_state = self.representation(observation)
        policy_logits, value = self.prediction(hidden_state)
        return hidden_state, policy_logits, value

    def recurrent_inference(
        self, hidden_state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Recurrent step: hidden state + action -> next hidden state, reward, policy, value."""
        next_hidden_state, reward = self.dynamics(hidden_state, action)
        policy_logits, value = self.prediction(next_hidden_state)
        return next_hidden_state, reward, policy_logits, value
