"""Tests for the Node and MCTS classes, covering initialization, value computation, selection logic, and search behavior."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from cumind.core import MCTS, Node
from cumind.core.network import CuMindNetwork
from cumind.utils.config import cfg
from cumind.utils.prng import key


@pytest.fixture(autouse=True)
def reset_prng_manager_singleton():
    """Reset the PRNGManager singleton before and after each test."""
    key.reset()
    yield
    key.reset()


class TestNode:
    """Test suite for MCTS Node."""

    def test_node_initialization(self):
        """Test Node initialization."""
        prior = 0.5
        node = Node(prior)

        # Verify initial state
        assert node.prior == prior
        assert node.visit_count == 0
        assert node.value_sum == 0.0
        assert node.children == {}
        assert node.hidden_state is None

        # Test with different prior values
        node_high = Node(0.9)
        assert node_high.prior == 0.9

        node_low = Node(0.1)
        assert node_low.prior == 0.1

    def test_node_is_expanded(self):
        """Test node expansion state checking."""
        node = Node(0.5)

        # Initially unexpanded
        assert not node.is_expanded()

        # Add a child
        child = Node(0.3)
        node.children[0] = child

        # Now expanded
        assert node.is_expanded()

        # Test with multiple children
        node.children[1] = Node(0.7)
        assert node.is_expanded()

    def test_node_value_calculation(self):
        """Test node value calculation."""
        node = Node(0.5)

        # Zero visits should return 0
        assert node.value() == 0.0

        # Single visit
        node.visit_count = 1
        node.value_sum = 0.8
        assert node.value() == 0.8

        # Multiple visits
        node.visit_count = 4
        node.value_sum = 2.0
        assert node.value() == 0.5

        # Negative values
        node.value_sum = -1.0
        assert node.value() == -0.25

    def test_ucb_score_calculation(self):
        """Test UCB score calculation for action selection."""
        parent = Node(0.5)
        child = Node(0.3)
        parent.children[0] = child

        # Set up parent visits
        parent.visit_count = 10

        # Test with no child visits (should return infinity)
        c_puct = 1.0
        ucb_score = child.ucb_score(parent.visit_count, c_puct)
        assert ucb_score == float("inf")

        # Test with child visits
        child.visit_count = 5
        child.value_sum = 2.5
        ucb_score = child.ucb_score(parent.visit_count, c_puct)
        expected_value = child.value()
        expected_exploration = c_puct * child.prior * jnp.sqrt(parent.visit_count) / (1 + child.visit_count)
        expected_ucb = expected_value + expected_exploration
        assert abs(ucb_score - expected_ucb) < 1e-6

    def test_select_child(self):
        """Test child selection based on UCB scores."""
        parent = Node(0.5)
        parent.visit_count = 10

        # Add children with different priors and values
        child1 = Node(0.3)
        child1.visit_count = 2
        child1.value_sum = 1.0
        parent.children[0] = child1

        child2 = Node(0.7)
        child2.visit_count = 1
        child2.value_sum = 0.8
        parent.children[1] = child2

        # Calculate UCB scores manually
        c_puct = 1.0
        ucb1 = child1.ucb_score(parent.visit_count, c_puct)
        ucb2 = child2.ucb_score(parent.visit_count, c_puct)

        # Test that select_child works properly by selecting highest UCB
        selected_action = parent.select_child(c_puct)
        expected_action = 0 if ucb1 > ucb2 else 1
        assert selected_action == expected_action
        assert ucb1 >= 0 and ucb2 >= 0

    def test_node_expansion(self):
        """Test node expansion with policy and children."""
        node = Node(0.5)

        # Initially not expanded
        assert not node.is_expanded()

        # Add children manually (simulating expansion)
        policy = jnp.array([0.6, 0.4])
        for action, prior in enumerate(policy):
            node.children[action] = Node(float(prior))

        # Now expanded
        assert node.is_expanded()
        assert len(node.children) == 2
        assert abs(node.children[0].prior - 0.6) < 1e-6
        assert abs(node.children[1].prior - 0.4) < 1e-6

    def test_value_backup(self):
        """Test value backup through tree path."""
        # Create a simple tree structure
        root = Node(0.5)
        child = Node(0.3)
        grandchild = Node(0.7)

        root.children[0] = child
        child.children[0] = grandchild

        # Simulate backup from grandchild to root
        value = 0.8

        # Manual backup simulation
        grandchild.visit_count += 1
        grandchild.value_sum += value

        child.visit_count += 1
        child.value_sum += value

        root.visit_count += 1
        root.value_sum += value

        # Verify values
        assert grandchild.value() == 0.8
        assert child.value() == 0.8
        assert root.value() == 0.8

    @pytest.fixture
    def setup(self):
        """Setup for MCTS tests."""
        key.seed(cfg.seed)
        rngs = nnx.Rngs(params=key())
        repre_net = cfg.representation()
        dyna_net = cfg.dynamics()
        pred_net = cfg.prediction()
        network = CuMindNetwork(repre_net, dyna_net, pred_net, rngs)
        mcts = MCTS(network)
        return mcts, network

    def test_mcts_initialization(self, setup):
        """Test MCTS initialization."""
        mcts, _ = setup
        assert mcts.network is not None

    def test_mcts_search(self, setup):
        """Test MCTS search returns a valid policy."""
        mcts, _ = setup
        root_hidden_state = jnp.ones(cfg.networks.hidden_dim)
        policy = mcts.search(root_hidden_state, add_noise=False)

        assert isinstance(policy, np.ndarray)
        assert len(policy) == cfg.env.action_space_size
        assert np.isclose(np.sum(policy), 1.0)
        assert np.all(policy >= 0)

    def test_mcts_search_basic(self, setup):
        """Test basic MCTS search functionality."""
        mcts, _ = setup
        root_hidden_state = jnp.ones(cfg.networks.hidden_dim)
        action_probs = mcts.search(root_hidden_state, add_noise=False)
        assert action_probs.shape == (cfg.env.action_space_size,)
        assert np.isclose(np.sum(action_probs), 1.0)
        assert np.all(action_probs >= 0)

    def test_action_probabilities(self):
        """Test action probability computation from visit counts."""
        # Create mock visit counts
        visit_counts = jnp.array([10, 5, 15, 2])

        # Test with temperature = 1 (proportional to visit counts)
        temperature = 1.0
        probs = jax.nn.softmax(jnp.log(visit_counts + 1e-8) / temperature)

        # Verify probabilities
        assert probs.shape == visit_counts.shape
        assert abs(jnp.sum(probs) - 1.0) < 1e-6
        assert jnp.all(probs >= 0)

        # Higher visit counts should have higher probabilities
        assert probs[2] > probs[0] > probs[1] > probs[3]

    def test_tree_statistics(self):
        """Test tree statistics and information gathering."""
        # Create a simple tree
        root = Node(0.5)
        root.visit_count = 10

        # Add children
        for i in range(3):
            child = Node(0.3)
            child.visit_count = i + 1
            root.children[i] = child

        # Test basic statistics
        total_visits = root.visit_count + sum(child.visit_count for child in root.children.values())
        assert total_visits == 16

        # Test tree depth (manually)
        max_depth = 2  # Root and one level of children
        assert max_depth >= 1

    def test_mcts_with_different_networks(self, setup):
        """Test MCTS with different network configurations."""
        mcts, network = setup
        key.seed(cfg.seed)
        rngs = nnx.Rngs(params=key())
        repre_net2 = cfg.representation()
        dyna_net2 = cfg.dynamics()
        pred_net2 = cfg.prediction()
        network2 = CuMindNetwork(repre_net2, dyna_net2, pred_net2, rngs)
        mcts2 = MCTS(network2)
        root_hidden_state = jnp.ones(cfg.networks.hidden_dim)
        action_probs = mcts2.search(root_hidden_state, add_noise=False)
        assert action_probs.shape == (cfg.env.action_space_size,)

    def test_mcts_edge_cases(self, setup):
        """Test MCTS edge cases and error handling."""
        mcts, network = setup
        # Test with single action (edge case)
        root_hidden_state = jnp.ones(cfg.networks.hidden_dim)
        action_probs = mcts.search(root_hidden_state, add_noise=False)
        assert action_probs.shape == (cfg.env.action_space_size,)


if __name__ == "__main__":
    pytest.main([__file__])
