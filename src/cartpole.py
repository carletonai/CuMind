"""Basic training example for CuMind with CartPole."""

import gymnasium as gym

from cumind.agent.agent import Agent
from cumind.agent.trainer import Trainer
from cumind.config import Config
from cumind.data.memory_buffer import ReplayBuffer
from cumind.utils.logger import Logger


def train_cartpole() -> None:
    """Train CuMind agent on the CartPole-v1 environment."""

    # Configuration for CartPole
    config = Config(
        hidden_dim=64,
        num_blocks=2,
        action_space_size=2,
        observation_shape=(4,),
        num_simulations=25,
        batch_size=32,
        learning_rate=3e-4,
        replay_buffer_size=10000,
        min_replay_size=200,
        td_steps=5,
        num_unroll_steps=3,
    )

    # Create environment
    env = gym.make("CartPole-v1")

    # Initialize components
    agent = Agent(config)
    memory_buffer = ReplayBuffer(capacity=config.replay_buffer_size)
    logger = Logger(log_dir="logs/cartpole")
    trainer = Trainer(agent, memory_buffer, config, logger)

    print("Starting CuMind training on CartPole...")
    print(f"Config: {config}")

    # Run the training loop
    trainer.run_training_loop(env, num_episodes=500, train_frequency=1)

    env.close()  # pyright: ignore
    logger.close()
    print("Training completed!")


if __name__ == "__main__":
    train_cartpole()
