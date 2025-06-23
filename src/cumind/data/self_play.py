"""Self-play runner for collecting training trajectories."""

from typing import Any, List

import numpy as np

from ..agent.agent import Agent
from ..config import Config
from .replay_buffer import ReplayBuffer


class SelfPlay:
    """Self-play runner: collects (o, a, r, o′) trajectories."""

    def __init__(self, config: Config, agent: Agent, replay_buffer: ReplayBuffer):
        """Initialize self-play runner.

        Args:
            config: CuMind configuration
            agent: Trained agent for self-play
            replay_buffer: Buffer to store collected trajectories
        """
        self.config = config
        self.agent = agent
        self.replay_buffer = replay_buffer
        self.episode_count = 0
        self.total_reward = 0.0

    def run_episode(self, environment: Any) -> List[Any]:
        """Run one self-play episode.

        Args:
            environment: Game environment

        Returns:
            List of (observation, action, reward, next_observation) tuples
        """
        trajectory = []
        observation, _ = environment.reset()  # Unpack observation and info
        done = False

        while not done:
            # Agent selects action
            action = self.agent.select_action(observation, training=True)

            # Environment step
            next_observation, reward, terminated, truncated, _ = environment.step(action)
            done = terminated or truncated

            # Store trajectory step
            trajectory.append({"observation": observation, "action": action, "reward": reward, "next_observation": next_observation, "done": done})

            observation = next_observation

        return trajectory

    def collect_trajectories(self, environment: Any, num_episodes: int) -> None:
        """Collect multiple trajectories from self-play.

        Args:
            environment: Game environment
            num_episodes: Number of episodes to collect
        """
        for _ in range(num_episodes):
            trajectory = self.run_episode(environment)
            self.replay_buffer.add(trajectory)
            self.episode_count += 1

            # Track statistics
            episode_reward = sum(step["reward"] for step in trajectory)
            self.total_reward += episode_reward

    def get_replay_buffer(self) -> ReplayBuffer:
        """Get the replay buffer with collected data.

        Returns:
            Replay buffer containing collected trajectories
        """
        return self.replay_buffer
