"""Basic training example for CuMind with CartPole."""

import gymnasium as gym
import numpy as np

from cumind.agent.agent import Agent
from cumind.config import Config
from cumind.data.memory_buffer import ReplayBuffer
from cumind.data.self_play import SelfPlay
from cumind.utils.logger import Logger


def train_cartpole() -> None:
    """Train CuMind agent on CartPole environment."""

    # Configuration for CartPole
    config = Config(hidden_dim=64, num_blocks=2, action_space_size=2, observation_shape=(4,), num_simulations=25, batch_size=16, learning_rate=3e-4, replay_buffer_size=1000, min_replay_size=100)

    # Create environment
    env = gym.make("CartPole-v1")

    # Initialize components
    agent = Agent(config)
    memory_buffer = ReplayBuffer(capacity=config.replay_buffer_size)
    self_play = SelfPlay(config, agent, memory_buffer)
    logger = Logger(log_dir="logs/cartpole")

    print("Starting CuMind training on CartPole...")
    print(f"Config: {config}")

    # Training loop
    num_episodes = 200
    train_frequency = 10  # Train every N episodes

    for episode in range(num_episodes):
        # Collect episode data
        episode_data = self_play.run_episode(env)
        episode_reward = sum(step["reward"] for step in episode_data)
        episode_length = len(episode_data)

        logger.log_scalar("episode_reward", episode_reward, episode)
        logger.log_scalar("episode_length", episode_length, episode)

        print(f"Episode {episode:3d}: Reward={episode_reward:6.1f}, Length={episode_length:3d}")

        # Train if we have enough data
        if episode % train_frequency == 0 and memory_buffer.is_ready(config.min_replay_size):
            # Sample batch and train
            batch = memory_buffer.sample(config.batch_size)
            losses = agent.train_step(batch)

            # Log training metrics
            for loss_name, loss_value in losses.items():
                logger.log_scalar(f"train/{loss_name}", loss_value, episode)

            print(f"  Training - Losses: {losses}")

        # Save checkpoint periodically
        if episode % 50 == 0 and episode > 0:
            checkpoint_path = f"checkpoints/cartpole_episode_{episode}.pkl"
            agent.save_checkpoint(checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")

    env.close()  # type: ignore
    logger.close()
    print("Training completed!")


if __name__ == "__main__":
    train_cartpole()
