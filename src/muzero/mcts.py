"""Monte Carlo Tree Search for MuZero."""

import math
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np
import torch

if TYPE_CHECKING:
    from .config import MuZeroConfig
    from .network import MuZeroNetwork


class Node:
    """MCTS node."""

    def __init__(self, prior: float):
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: Dict[int, "Node"] = {}
        self.hidden_state: Optional[torch.Tensor] = None

    def is_expanded(self) -> bool:
        return len(self.children) > 0

    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb_score(self, parent_visit_count: int, c_puct: float) -> float:
        """Calculate UCB score for node selection."""
        if self.visit_count == 0:
            return float("inf")

        exploration = (
            c_puct * self.prior * math.sqrt(parent_visit_count) / (1 + self.visit_count)
        )
        return self.value() + exploration

    def select_child(self, c_puct: float) -> int:
        """Select child with highest UCB score."""
        return max(
            self.children.keys(),
            key=lambda action: self.children[action].ucb_score(
                self.visit_count, c_puct
            ),
        )

    def expand(
        self, actions: List[int], priors: torch.Tensor, hidden_state: torch.Tensor
    ) -> None:
        """Expand node with children."""
        self.hidden_state = hidden_state
        for action, prior in zip(actions, priors):
            self.children[action] = Node(prior.item())

    def backup(self, value: float) -> None:
        """Backup value through the tree."""
        self.visit_count += 1
        self.value_sum += value


class MCTS:
    """Monte Carlo Tree Search."""

    def __init__(self, config: "MuZeroConfig"):
        self.config = config

    def search(
        self, network: "MuZeroNetwork", root_hidden_state: torch.Tensor
    ) -> np.ndarray:
        """Run MCTS and return action probabilities."""
        # Create root node
        with torch.no_grad():
            policy_logits, value = network.prediction(root_hidden_state.unsqueeze(0))
            policy_probs = torch.softmax(policy_logits, dim=-1).squeeze(0)

        actions = list(range(self.config.action_space_size))
        root = Node(prior=0.0)
        root.expand(actions, policy_probs, root_hidden_state)

        # Add Dirichlet noise to root
        self._add_exploration_noise(root)

        # Run simulations
        for _ in range(self.config.num_simulations):
            self._simulate(network, root)

        # Extract action probabilities
        visit_counts = np.array(
            [
                root.children[action].visit_count if action in root.children else 0
                for action in actions
            ]
        )

        # Temperature-based selection (could be added to config)
        if visit_counts.sum() == 0:
            # If no visits, return uniform distribution
            action_probs = np.ones(len(actions)) / len(actions)
        else:
            action_probs = visit_counts / visit_counts.sum()
        return action_probs

    def _simulate(self, network: "MuZeroNetwork", root: Node) -> None:
        """Run one MCTS simulation."""
        path = []
        node = root

        # Selection phase
        while node.is_expanded():
            action = node.select_child(self.config.c_puct)
            path.append((node, action))
            node = node.children[action]

        # Expansion and evaluation phase
        parent, action = path[-1] if path else (None, 0)

        if parent is not None and parent.hidden_state is not None:
            # Get hidden state from parent and action
            with torch.no_grad():
                hidden_state, reward, policy_logits, value = (
                    network.recurrent_inference(
                        parent.hidden_state.unsqueeze(0), torch.tensor([action])
                    )
                )
                policy_probs = torch.softmax(policy_logits, dim=-1).squeeze(0)

            # Expand node
            actions = list(range(self.config.action_space_size))
            node.expand(actions, policy_probs, hidden_state.squeeze(0))
            node_value = value.item()
        else:
            # Root node case
            if node.hidden_state is not None:
                with torch.no_grad():
                    policy_logits, value = network.prediction(
                        node.hidden_state.unsqueeze(0)
                    )
                node_value = value.item()
            else:
                node_value = 0.0

        # Backup phase
        for node, _ in reversed(path):
            node.backup(node_value)
            node_value *= self.config.discount  # Apply discount

    def _add_exploration_noise(self, root: Node) -> None:
        """Add Dirichlet noise to root node for exploration."""
        actions = list(root.children.keys())
        noise = np.random.dirichlet([self.config.dirichlet_alpha] * len(actions))

        for action, noise_value in zip(actions, noise):
            child = root.children[action]
            child.prior = (
                1 - self.config.exploration_fraction
            ) * child.prior + self.config.exploration_fraction * noise_value
