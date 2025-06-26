"""Self-play runner for collecting training data samples."""

from typing import Any, Dict, List

import numpy as np

from ..agent.agent import Agent
from ..config import Config
from ..utils.logger import log
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
        log.info("Initializing SelfPlay runner.")
        self.config = config
        self.agent = agent
        self.memory_buffer = memory_buffer

    def run_episode(self, environment: Any) -> tuple[float, int, List[Dict[str, Any]]]:
        """Run one self-play episode and collect data.

        Args:
            environment: The game environment.

        Returns:
            A tuple containing total reward, episode length, and a list of step data dictionaries.
        """
        log.debug("Starting new self-play episode.")
        episode_data = []
        observation, _ = environment.reset()
        done = False

        total_reward = 0.0
        episode_steps = 0

        while not done:
            # Agent selects action and gets MCTS policy
            action, policy = self.agent.select_action(observation, training=True)

            # Environment step
            next_observation, reward, terminated, truncated, _ = environment.step(action)
            done = terminated or truncated

            # Store step data
            episode_data.append({"observation": observation, "action": action, "reward": reward, "policy": policy, "done": done})

            observation = next_observation

            total_reward += reward
            episode_steps += 1

        log.debug(f"Episode finished. Total reward: {total_reward}, Steps: {episode_steps}.")
        self.memory_buffer.add(episode_data)
        log.debug(f"Added episode data to memory buffer. Buffer size: {len(self.memory_buffer)}.")
        return total_reward, episode_steps, episode_data

    def collect_samples(self, environment: Any, num_episodes: int) -> None:
        """Collect data from multiple self-play episodes.

        Args:
            environment: The game environment.
            num_episodes: Number of episodes to run.
        """
        log.info(f"Starting sample collection for {num_episodes} episodes.")
        for i in range(num_episodes):
            log.debug(f"Running episode {i + 1}/{num_episodes}.")
            self.run_episode(environment)
        log.info(f"Finished collecting samples for {num_episodes} episodes.")

    def get_memory_buffer(self) -> MemoryBuffer:
        """Get the memory buffer with collected data.

        Returns:
            Memory buffer containing collected data samples
        """
        return self.memory_buffer
