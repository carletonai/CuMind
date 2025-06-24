"""Self-play runner for collecting training data samples."""

from typing import Any, List

import numpy as np

from ..agent.agent import Agent
from ..config import Config
from .memory_buffer import MemoryBuffer


class SelfPlay:
    """Self-play runner: collects (s, a, r, s′) data samples."""

    def __init__(self, config: Config, agent: Agent, memory_buffer: MemoryBuffer):
        """Initialize self-play runner.

        Args:
            config: CuMind configuration
            agent: Trained agent for self-play
            memory_buffer: Buffer to store collected data samples
        """
        self.config = config
        self.agent = agent
        self.memory_buffer = memory_buffer
        self.episode_count = 0
        self.total_reward = 0.0

    def run_episode(self, environment: Any) -> List[Any]:
        """Run one self-play episode.

        Args:
            environment: Game environment

        Returns:
            List of (observation, action, reward, next_observation) data samples
        """
        episode_data = []
        observation, _ = environment.reset()  # Unpack observation and info
        done = False

        while not done:
            # Agent selects action
            action = self.agent.select_action(observation, training=True)

            # Environment step
            next_observation, reward, terminated, truncated, _ = environment.step(action)
            done = terminated or truncated

            # Store episode step
            episode_data.append({"observation": observation, "action": action, "reward": reward, "next_observation": next_observation, "done": done})

            observation = next_observation

        return episode_data

    def collect_samples(self, environment: Any, num_episodes: int) -> None:
        """Collect multiple data samples from self-play.

        Args:
            environment: Game environment
            num_episodes: Number of episodes to collect
        """
        for _ in range(num_episodes):
            episode_data = self.run_episode(environment)
            self.memory_buffer.add(episode_data)
            self.episode_count += 1

            # Track statistics
            episode_reward = sum(step["reward"] for step in episode_data)
            self.total_reward += episode_reward

    def get_memory_buffer(self) -> MemoryBuffer:
        """Get the memory buffer with collected data.

        Returns:
            Memory buffer containing collected data samples
        """
        return self.memory_buffer
