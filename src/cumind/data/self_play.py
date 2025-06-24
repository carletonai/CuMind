"""Self-play runner for collecting training data samples."""

from typing import Any, Dict, List

import numpy as np

from ..agent.agent import Agent
from ..config import Config
from .memory_buffer import MemoryBuffer


class SelfPlay:
    """Self-play runner: collects game data and stores it in a buffer."""

    def __init__(self, config: Config, agent: Agent, memory_buffer: MemoryBuffer):
        """Initialize self-play runner.

        Args:
            config: CuMind configuration.
            agent: Agent for self-play.
            memory_buffer: Buffer to store collected data.
        """
        self.config = config
        self.agent = agent
        self.memory_buffer = memory_buffer

    def run_episode(self, environment: Any) -> List[Dict[str, Any]]:
        """Run one self-play episode and collect data.

        Args:
            environment: The game environment.

        Returns:
            A list of dictionaries, where each dictionary contains data for a single step.
        """
        episode_data = []
        observation, _ = environment.reset()
        done = False

        while not done:
            # Agent selects action and gets MCTS policy
            action, policy = self.agent.select_action(observation, training=True)

            # Environment step
            next_observation, reward, terminated, truncated, _ = environment.step(action)
            done = terminated or truncated

            # Store step data
            episode_data.append({"observation": observation, "action": action, "reward": reward, "policy": policy, "done": done})

            observation = next_observation

        self.memory_buffer.add(episode_data)
        return episode_data

    def collect_samples(self, environment: Any, num_episodes: int) -> None:
        """Collect data from multiple self-play episodes.

        Args:
            environment: The game environment.
            num_episodes: Number of episodes to run.
        """
        for _ in range(num_episodes):
            self.run_episode(environment)

    def get_memory_buffer(self) -> MemoryBuffer:
        """Get the memory buffer with collected data.

        Returns:
            Memory buffer containing collected data samples
        """
        return self.memory_buffer
