"""Monte Carlo Tree Search for CuMind."""

import math
from typing import TYPE_CHECKING, Dict, List, Optional

import chex
import jax
import jax.numpy as jnp
import numpy as np

from ..utils.logger import log

if TYPE_CHECKING:
    from ..config import Config
    from .network import CuMindNetwork


class Node:
    """MCTS node."""

    def __init__(self, prior: float):
        """Initialize MCTS node with prior probability.

        Args:
            prior: Prior probability for this node
        """
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: Dict[int, "Node"] = {}
        self.hidden_state: Optional[chex.Array] = None

    def is_expanded(self) -> bool:
        """Check if node has children.

        Returns:
            True if node has been expanded with children
        """
        return len(self.children) > 0

    def value(self) -> float:
        """Calculate average value from visits.

        Returns:
            Average value (value_sum / visit_count) or 0 if no visits
        """
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb_score(self, parent_visit_count: int, c_puct: float) -> float:
        """Calculate UCB score for node selection.

        Args:
            parent_visit_count: Number of times parent was visited
            c_puct: Exploration constant

        Returns:
            UCB score combining exploitation and exploration
        """
        if self.visit_count == 0:
            return float("inf")

        exploration_term = c_puct * self.prior * math.sqrt(parent_visit_count) / (1 + self.visit_count)
        return self.value() + exploration_term

    def select_child(self, c_puct: float) -> int:
        """Select child with highest UCB score.

        Args:
            c_puct: Exploration constant

        Returns:
            Action index of child with max UCB score
        """
        return max(self.children.keys(), key=lambda action: self.children[action].ucb_score(self.visit_count, c_puct))

    def expand(self, actions: List[int], priors: chex.Array, hidden_state: chex.Array) -> None:
        """Expand node with children.

        Args:
            actions: List of possible actions
            priors: Prior probabilities for each action
            hidden_state: Hidden state for this node
        """
        self.hidden_state = hidden_state
        priors_array = jnp.asarray(priors, dtype=jnp.float32)
        for action, prior in zip(actions, jnp.asarray(priors_array)):
            self.children[action] = Node(float(prior))

    def backup(self, value: float) -> None:
        """Backup value through the tree.

        Args:
            value: Value to add to this node
        """
        self.visit_count += 1
        self.value_sum += value


class MCTS:
    """Monte Carlo Tree Search."""

    def __init__(self, network: "CuMindNetwork", config: "Config"):
        """Initialize MCTS with a network and configuration.

        Args:
            network: The CuMind network for model-based rollouts.
            config: CuMind configuration with MCTS parameters.
        """
        log.info("Initializing MCTS...")
        self.network = network
        self.config = config

    def search(self, root_hidden_state: chex.Array, add_noise: bool = True) -> np.ndarray:
        """Run MCTS and return action probabilities.

        Args:
            root_hidden_state: Hidden state of the root node.
            add_noise: If True, add exploration noise to the root's priors.

        Returns:
            Action probability distribution.
        """
        log.debug(f"Starting MCTS search with {self.config.num_simulations} simulations. Noise: {add_noise}")
        root_hidden_state_batched = jnp.expand_dims(root_hidden_state, 0)
        policy_logits, _ = self.network.prediction_network(root_hidden_state_batched)
        priors = jax.nn.softmax(policy_logits, axis=-1)[0]

        # Create root node and expand
        root = Node(1.0)
        actions = list(range(self.config.action_space_size))
        root.expand(actions, priors, root_hidden_state)

        if add_noise:
            self._add_exploration_noise(root)

        log.debug(f"Running {self.config.num_simulations} simulations...")
        for _ in range(self.config.num_simulations):
            self._simulate(root)

        # Extract visit counts and normalize
        visit_counts = np.zeros(self.config.action_space_size)
        for action, child in root.children.items():
            visit_counts[action] = child.visit_count

        # Normalize
        total_visits = np.sum(visit_counts)
        if total_visits == 0:
            log.warning("MCTS search resulted in zero visits. Returning uniform probabilities.")
            return np.ones(self.config.action_space_size) / self.config.action_space_size

        action_probs = visit_counts / total_visits
        log.debug(f"MCTS search complete. Action probabilities: {action_probs}")
        return action_probs

    def _simulate(self, root: Node) -> None:
        """Run one MCTS simulation from the root.

        Args:
            root: The root node of the search tree.
        """
        # Selection: traverse tree using UCB until a leaf is reached
        log.debug("MCTS simulation: Selection phase.")
        path = []
        node = root

        while node.is_expanded():
            action = node.select_child(self.config.c_puct)
            path.append((node, action))
            node = node.children[action]

        leaf_value = 0.0
        log.debug("MCTS simulation: Expansion and evaluation phase.")
        if node.hidden_state is None:
            log.debug("MCTS simulation: Leaf node has no hidden state, computing it.")
            if len(path) > 0:
                parent_node, action = path[-1]
                if parent_node.hidden_state is not None:
                    # Run dynamics to get the next state for the leaf
                    next_state, _ = self.network.dynamics_network(jnp.expand_dims(parent_node.hidden_state, 0), jnp.array([action]))
                    node.hidden_state = jnp.asarray(next_state)[0]
            else:
                log.debug("MCTS simulation: Node is root, using its hidden state.")
                node.hidden_state = root.hidden_state

        # Evaluate leaf and expand it
        if node.hidden_state is not None:
            hidden_state_expanded = jnp.expand_dims(node.hidden_state, 0)
            policy_logits, value = self.network.prediction_network(hidden_state_expanded)
            priors = jax.nn.softmax(policy_logits, axis=-1)
            priors_array = np.asarray(priors[0], dtype=np.float32)
            leaf_value = float(jnp.asarray(value)[0, 0])

            actions = list(range(self.config.action_space_size))
            node.expand(actions, priors_array, jnp.asarray(node.hidden_state))
            log.debug(f"MCTS simulation: Expanded leaf node with value {leaf_value:.4f}.")

        # Backup: propagate the leaf's value up the path
        log.debug(f"MCTS simulation: Backup phase with value {leaf_value:.4f}.")
        for node, _ in reversed(path):
            node.backup(leaf_value)
        root.backup(leaf_value)

    def _add_exploration_noise(self, root: Node) -> None:
        """Add Dirichlet noise to the root node's priors for exploration.

        Args:
            root: The root node to add noise to.
        """
        log.debug("Adding exploration noise to root node.")
        if not root.children:
            log.warning("Cannot add exploration noise to a root node with no children.")
            return

        # Sample Dirichlet noise
        num_actions = len(root.children)
        noise = np.random.dirichlet([self.config.dirichlet_alpha] * num_actions)

        # Mix with existing priors
        for i, (_, child) in enumerate(root.children.items()):
            child.prior = child.prior * (1 - self.config.exploration_fraction) + noise[i] * self.config.exploration_fraction
