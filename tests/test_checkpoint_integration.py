#!/usr/bin/env python3
"""Test script to verify memory buffer checkpoint integration."""

import os
import tempfile

from cumind.agent.agent import Agent
from cumind.data.memory import MemoryBuffer, PrioritizedMemoryBuffer, TreeBuffer
from cumind.utils.checkpoint import AgentState, load_checkpoint, save_checkpoint


def test_memory_checkpoint_integration():
    """Test that memory buffer state can be saved and loaded correctly."""
    # Initialize configuration

    os.environ["CUMIND_ENV"] = "test"

    print("Testing memory buffer checkpoint integration...")

    # Test MemoryBuffer
    print("\n1. Testing MemoryBuffer:")
    buffer1 = MemoryBuffer(100)

    # Add some sample data
    for i in range(5):
        sample = [{"observation": i, "action": i % 2, "reward": float(i)}]
        buffer1.add(sample)

    print(f"   Original buffer size: {len(buffer1)}")

    # Save state
    state1 = buffer1.save_state()
    print(f"   Saved state keys: {list(state1.keys())}")

    # Create new buffer and load state
    buffer2 = MemoryBuffer(100)
    buffer2.load_state(state1)
    print(f"   Loaded buffer size: {len(buffer2)}")

    assert len(buffer1) == len(buffer2), "Buffer sizes don't match"
    assert buffer1.capacity == buffer2.capacity, "Capacities don't match"
    print("   ✓ MemoryBuffer state save/load works correctly")

    # Test PrioritizedMemoryBuffer
    print("\n2. Testing PrioritizedMemoryBuffer:")
    priority_buffer1 = PrioritizedMemoryBuffer(100, alpha=0.6, epsilon=1e-6, beta=0.4)

    # Add some sample data
    for i in range(5):
        sample = [{"observation": i, "action": i % 2, "reward": float(i)}]
        priority_buffer1.add(sample)

    print(f"   Original buffer size: {len(priority_buffer1)}")

    # Save state
    priority_state1 = priority_buffer1.save_state()
    print(f"   Saved state keys: {list(priority_state1.keys())}")

    # Create new buffer and load state
    priority_buffer2 = PrioritizedMemoryBuffer(100, alpha=0.6, epsilon=1e-6, beta=0.4)
    priority_buffer2.load_state(priority_state1)
    print(f"   Loaded buffer size: {len(priority_buffer2)}")

    assert len(priority_buffer1) == len(priority_buffer2), "Buffer sizes don't match"
    assert priority_buffer1.alpha == priority_buffer2.alpha, "Alpha values don't match"
    assert priority_buffer1.max_priority == priority_buffer2.max_priority, "Max priorities don't match"
    print("   ✓ PrioritizedMemoryBuffer state save/load works correctly")

    # Test TreeBuffer
    print("\n3. Testing TreeBuffer:")
    tree_buffer1 = TreeBuffer(100, alpha=0.6, epsilon=1e-6)

    # Add some sample data
    for i in range(5):
        sample = [{"observation": i, "action": i % 2, "reward": float(i)}]
        tree_buffer1.add(sample)

    print(f"   Original buffer size: {len(tree_buffer1)}")

    # Save state
    tree_state1 = tree_buffer1.save_state()
    print(f"   Saved state keys: {list(tree_state1.keys())}")

    # Create new buffer and load state
    tree_buffer2 = TreeBuffer(100, alpha=0.6, epsilon=1e-6)
    tree_buffer2.load_state(tree_state1)
    print(f"   Loaded buffer size: {len(tree_buffer2)}")

    assert len(tree_buffer1) == len(tree_buffer2), "Buffer sizes don't match"
    assert tree_buffer1.alpha == tree_buffer2.alpha, "Alpha values don't match"
    assert tree_buffer1.n_entries == tree_buffer2.n_entries, "Entry counts don't match"
    print("   ✓ TreeBuffer state save/load works correctly")

    # Test full checkpoint integration
    print("\n4. Testing full checkpoint integration:")

    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_path = os.path.join(temp_dir, "test_checkpoint.pkl")

        # Create agent and memory buffer
        agent = Agent()
        memory = MemoryBuffer(100)

        # Add some data to memory
        for i in range(3):
            sample = [{"observation": i, "action": i % 2, "reward": float(i)}]
            memory.add(sample)

        print(f"   Memory buffer size before checkpoint: {len(memory)}")

        # Create agent state with memory
        agent_state: AgentState = agent.save_state()
        agent_state["memory_state"] = memory.save_state()

        # Save checkpoint
        save_checkpoint(agent_state, checkpoint_path)
        print(f"   Checkpoint saved to: {checkpoint_path}")

        # Load checkpoint
        loaded_data = load_checkpoint(checkpoint_path)
        loaded_agent_state = loaded_data["state"]

        # Create new memory buffer and load state
        new_memory = MemoryBuffer(100)
        if loaded_agent_state["memory_state"]:
            new_memory.load_state(loaded_agent_state["memory_state"])

        print(f"   Memory buffer size after loading: {len(new_memory)}")

        assert len(memory) == len(new_memory), "Memory sizes don't match after checkpoint"
        print("   ✓ Full checkpoint integration works correctly")

    print("\n✅ All memory buffer checkpoint integration tests passed!")


if __name__ == "__main__":
    test_memory_checkpoint_integration()
