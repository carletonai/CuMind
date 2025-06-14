"""MuZero agent implementation."""

from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as functional

from .config import MuZeroConfig
from .mcts import MCTS
from .network import MuZeroNetwork


class MuZeroAgent:
    """MuZero agent for training and inference."""

    def __init__(self, config: MuZeroConfig):
        self.config = config
        self.network = MuZeroNetwork(
            observation_shape=config.observation_shape,
            action_space_size=config.action_space_size,
            hidden_dim=config.hidden_dim,
            num_blocks=config.num_blocks,
        )
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.mcts = MCTS(config)

    def select_action(self, observation: np.ndarray, training: bool = True) -> int:
        """Select action using MCTS."""
        observation_tensor = torch.FloatTensor(torch.FloatTensor(observation).unsqueeze(0))

        with torch.no_grad():
            hidden_state, policy_logits, value = self.network.initial_inference(observation_tensor)

        # Run MCTS
        action_probs = self.mcts.search(self.network, hidden_state.squeeze(0))

        if training:
            # Sample from distribution during training
            action = int(np.random.choice(len(action_probs), p=action_probs))
        else:
            # Take best action during evaluation
            action = int(np.argmax(action_probs))

        return action

    def train_step(self, batch: List[Any]) -> Dict[str, float]:
        """Perform one training step."""
        # Extract batch data
        observations, actions, targets = self._prepare_batch(batch)

        # Forward pass
        losses = self._compute_losses(observations, actions, targets)

        # Backward pass
        loss_values = list(losses.values())
        total_loss = loss_values[0]
        for loss in loss_values[1:]:
            total_loss = total_loss + loss
        self.optimizer.zero_grad()
        total_loss.backward()  # type: ignore[no-untyped-call]
        self.optimizer.step()

        # Return loss info
        return {k: v.item() for k, v in losses.items()}

    def _prepare_batch(
        self, batch: List[Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Prepare batch data for training."""
        # This would be implemented based on your replay buffer format
        # For now, return dummy tensors
        batch_size = self.config.batch_size
        obs_shape = self.config.observation_shape

        observations = torch.randn(batch_size, *obs_shape)
        actions = torch.randint(
            0, self.config.action_space_size, (batch_size, self.config.num_unroll_steps)
        )

        targets = {
            "values": torch.randn(batch_size, self.config.num_unroll_steps + 1),
            "rewards": torch.randn(batch_size, self.config.num_unroll_steps),
            "policies": torch.randn(
                batch_size,
                self.config.num_unroll_steps + 1,
                self.config.action_space_size,
            ),
        }

        return observations, actions, targets

    def _compute_losses(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute MuZero losses."""
        # Remove unused variable
        # Initial inference
        hidden_state, policy_logits, value = self.network.initial_inference(observations)

        # Collect predictions
        value_preds = [value.squeeze(-1)]
        policy_preds = [policy_logits]
        reward_preds = []

        # Unroll dynamics
        for k in range(self.config.num_unroll_steps):
            hidden_state, reward, policy_logits, value = self.network.recurrent_inference(
                hidden_state, actions[:, k]
            )
            value_preds.append(value.squeeze(-1))
            policy_preds.append(policy_logits)
            reward_preds.append(reward.squeeze(-1))

        # Compute losses
        value_loss = sum(
            functional.mse_loss(pred, target)
            for pred, target in zip(value_preds, targets["values"].unbind(1))
        )

        policy_loss = sum(
            functional.cross_entropy(pred, target.argmax(-1))
            for pred, target in zip(policy_preds, targets["policies"].unbind(1))
        )

        reward_loss = sum(
            functional.mse_loss(pred, target)
            for pred, target in zip(reward_preds, targets["rewards"].unbind(1))
        )

        return {
            "value_loss": value_loss / torch.tensor(len(value_preds), dtype=torch.float),
            "policy_loss": policy_loss / torch.tensor(len(policy_preds), dtype=torch.float),
            "reward_loss": (reward_loss / torch.tensor(len(reward_preds), dtype=torch.float))
            if reward_preds
            else torch.tensor(0.0),
        }

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint."""
        torch.save(
            {
                "network_state_dict": self.network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config,
            },
            path,
        )

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path)
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
