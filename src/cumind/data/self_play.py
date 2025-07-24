"""Self-play runner for collecting training data samples."""

from typing import Any, Dict, List

from cumind.agent.agent import Agent
from cumind.data.memory import Memory
from cumind.utils.logger import log


class SelfPlay:
    """Self-play runner: collects game data and stores it in a buffer."""

    def __init__(self, agent: Agent, memory: Memory):
        """Initialize self-play runner.

        Args:
            agent: Agent for self-play.
            memory: Buffer to store collected data.
        """
        log.info("Initializing SelfPlay runner.")
        self.agent = agent
        self.memory = memory

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

            # Store data
            episode_data.append({"observation": observation, "action": action, "reward": reward, "policy": policy, "done": done})

            observation = next_observation

            total_reward += reward
            episode_steps += 1

        log.debug(f"Episode finished. Total reward: {total_reward}, Steps: {episode_steps}.")
        self.memory.add(episode_data)
        log.debug(f"Added episode data to memory buffer. Buffer size: {len(self.memory)}.")
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

    def get_memory(self) -> Memory:
        """Get the memory buffer with collected data.

        Returns:
            Memory buffer containing collected data samples
        """
        return self.memory
