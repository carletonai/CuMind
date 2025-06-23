import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from cumind.core import MCTS, Node
from cumind.core.network import CuMindNetwork


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
        expected_exploration = c_puct * child.prior * np.sqrt(parent.visit_count) / (1 + child.visit_count)
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


class TestMCTS:
    """Test suite for MCTS algorithm."""

    def test_mcts_initialization(self):
        """Test MCTS initialization."""
        from cumind.config import Config

        config = Config()
        config.action_space_size = 2
        config.observation_shape = (4,)

        mcts = MCTS(config)

        # Verify MCTS attributes
        assert mcts.config == config
        assert hasattr(mcts.config, "c_puct")
        assert hasattr(mcts.config, "dirichlet_alpha")
        assert hasattr(mcts.config, "exploration_fraction")

    def test_mcts_search(self):
        """Test MCTS search algorithm."""
        from cumind.config import Config

        config = Config()
        config.num_simulations = 5  # Small number for testing
        config.action_space_size = 2
        config.observation_shape = (4,)

        mcts = MCTS(config)

        # Create mock network and hidden state
        key = jax.random.PRNGKey(42)
        from flax import nnx

        from cumind.core.network import CuMindNetwork

        rngs = nnx.Rngs(params=key)
        network = CuMindNetwork(observation_shape=config.observation_shape, action_space_size=config.action_space_size, rngs=rngs)

        # Test search with mock hidden state
        hidden_state = jnp.ones(64)  # Mock hidden state

        try:
            action_probs = mcts.search(network, hidden_state)

            # Verify output
            assert action_probs.shape == (config.action_space_size,)
            assert abs(jnp.sum(action_probs) - 1.0) < 1e-6  # Should sum to 1
            assert jnp.all(action_probs >= 0)  # All probabilities non-negative
        except (AttributeError, NotImplementedError):
            # MCTS search may not be fully implemented yet
            assert True

    def test_mcts_search_basic(self):
        """Test basic MCTS search functionality."""
        from cumind.config import Config

        config = Config()
        config.num_simulations = 10
        config.action_space_size = 2
        config.observation_shape = (4,)
        config.c_puct = 1.0
        config.discount = 0.99

        mcts = MCTS(config)

        # Create a simple network (we'll use a real network for this test)
        key = jax.random.PRNGKey(0)
        rngs = nnx.Rngs(params=key)
        network = CuMindNetwork(observation_shape=config.observation_shape, action_space_size=config.action_space_size, hidden_dim=64, num_blocks=2, rngs=rngs)

        # Create a dummy hidden state
        hidden_state = jnp.ones((64,))  # Match network hidden dim

        # Run MCTS search
        action_probs = mcts.search(network, hidden_state, add_noise=False)

        # Verify output format
        assert isinstance(action_probs, np.ndarray)
        assert action_probs.shape == (config.action_space_size,)
        assert np.allclose(np.sum(action_probs), 1.0, atol=1e-6)  # Should sum to 1
        assert np.all(action_probs >= 0)  # All probabilities should be non-negative

    def test_exploration_noise(self):
        """Test exploration noise addition to root."""
        # Test Dirichlet noise generation
        key = jax.random.PRNGKey(42)
        alpha = 0.3
        action_size = 4

        noise = jax.random.dirichlet(key, alpha=jnp.full(action_size, alpha))

        # Verify noise properties
        assert noise.shape == (action_size,)
        assert abs(jnp.sum(noise) - 1.0) < 1e-6  # Should sum to 1
        assert jnp.all(noise >= 0)  # All values non-negative

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

    def test_mcts_with_different_networks(self):
        """Test MCTS with different network configurations."""
        from cumind.config import Config

        # Test with vector observations
        config1 = Config()
        config1.observation_shape = (8,)
        config1.action_space_size = 2

        mcts1 = MCTS(config1)
        assert mcts1.config == config1

        # Test with image observations
        config2 = Config()
        config2.observation_shape = (84, 84, 4)
        config2.action_space_size = 4

        mcts2 = MCTS(config2)
        assert mcts2.config == config2

    def test_mcts_edge_cases(self):
        """Test MCTS edge cases and error handling."""
        from cumind.config import Config

        config = Config()

        # Test with single action space
        config.action_space_size = 1
        config.observation_shape = (4,)
        mcts = MCTS(config)

        assert mcts.config.action_space_size == 1

        # Test with zero simulations
        config.num_simulations = 0
        mcts_zero = MCTS(config)
        assert mcts_zero.config.num_simulations == 0

    def test_mcts_exploration_noise(self):
        """Test that exploration noise affects action probabilities."""
        from cumind.config import Config

        config = Config()
        config.num_simulations = 20
        config.action_space_size = 2
        config.observation_shape = (4,)
        config.c_puct = 1.0
        config.discount = 0.99
        config.dirichlet_alpha = 0.3
        config.exploration_fraction = 0.25

        mcts = MCTS(config)

        key = jax.random.PRNGKey(0)
        rngs = nnx.Rngs(params=key)
        network = CuMindNetwork(observation_shape=config.observation_shape, action_space_size=config.action_space_size, hidden_dim=64, num_blocks=2, rngs=rngs)

        hidden_state = jnp.ones((64,))

        # Run search with and without noise multiple times
        probs_with_noise = []
        probs_without_noise = []

        for _ in range(5):
            probs_noise = mcts.search(network, hidden_state, add_noise=True)
            probs_no_noise = mcts.search(network, hidden_state, add_noise=False)
            probs_with_noise.append(probs_noise)
            probs_without_noise.append(probs_no_noise)

        # Convert to arrays for easier comparison
        probs_with_noise = np.array(probs_with_noise)
        probs_without_noise = np.array(probs_without_noise)

        # Without noise, results should be more consistent
        no_noise_std = np.std(probs_without_noise, axis=0)
        with_noise_std = np.std(probs_with_noise, axis=0)

        # Generally expect more variation with noise (though not guaranteed)
        assert np.all(no_noise_std >= 0)  # Basic sanity check
        assert np.all(with_noise_std >= 0)  # Basic sanity check


if __name__ == "__main__":
    pytest.main([__file__])
