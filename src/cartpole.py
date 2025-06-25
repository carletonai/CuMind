"""Basic training example for CuMind with CartPole."""

import gymnasium as gym

from cumind.agent.agent import Agent
from cumind.agent.trainer import Trainer
from cumind.config import Config
from cumind.data.memory_buffer import ReplayBuffer
from cumind.utils.logger import Logger


def train_cartpole() -> None:
    config = Config(
        hidden_dim=128,
        num_blocks=2,
        action_space_size=2,
        observation_shape=(4,),
        num_simulations=25,
        batch_size=32,
        learning_rate=1e-3,
        replay_buffer_size=1000,
        min_replay_size=5,
        td_steps=5,
        num_unroll_steps=3,
    )

    env = gym.make("CartPole-v1")

    agent = Agent(config)
    memory_buffer = ReplayBuffer(capacity=config.replay_buffer_size)
    logger = Logger(log_dir="logs/cartpole")
    trainer = Trainer(agent, memory_buffer, config, logger)

    print("Starting CuMind training on CartPole...")
    print(f"Config: {config}")

    trainer.run_training_loop(env, num_episodes=5000, train_frequency=10)

    env.close()
    logger.close()
    print("Training completed!")


if __name__ == "__main__":
    train_cartpole()
