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

        Implementation:
            - Store config, agent, and replay buffer references
            - Initialize episode counter and statistics
        """
        # Branch: feature/self-play-init
        raise NotImplementedError("SelfPlay.__init__ needs to be implemented")

    def run_episode(self, environment: Any) -> List[Any]:
        """Run one self-play episode.

        Args:
            environment: Game environment

        Returns:
            List of (observation, action, reward, next_observation) tuples

        Implementation:
            - Reset environment and get initial observation
            - Use agent to select actions until episode ends
            - Collect trajectory data at each step
            - Return complete episode trajectory
        """
        # Branch: feature/self-play-episode
        raise NotImplementedError("SelfPlay.run_episode needs to be implemented")

    def collect_trajectories(self, environment: Any, num_episodes: int) -> None:
        """Collect multiple trajectories from self-play.

        Args:
            environment: Game environment
            num_episodes: Number of episodes to collect

        Implementation:
            - Loop for num_episodes iterations
            - Call run_episode() for each iteration
            - Add each trajectory to replay buffer
            - Track collection statistics
        """
        # Branch: feature/collect-trajectories
        raise NotImplementedError("SelfPlay.collect_trajectories needs to be implemented")

    def get_replay_buffer(self) -> ReplayBuffer:
        """Get the replay buffer with collected data.

        Returns:
            Replay buffer containing collected trajectories

        Implementation:
            - Return reference to internal replay buffer
        """
        # Branch: feature/get-replay-buffer
        raise NotImplementedError("SelfPlay.get_replay_buffer needs to be implemented")
