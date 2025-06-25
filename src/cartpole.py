"""Basic training example for CuMind with CartPole."""

import os
from datetime import datetime

import gymnasium as gym

from cumind.agent.agent import Agent
from cumind.agent.trainer import Trainer
from cumind.config import Config
from cumind.data.memory_buffer import ReplayBuffer
from cumind.utils.checkpoint import load_checkpoint
from cumind.utils.logger import Logger


def train(config: Config, run_name: str = "cartpole") -> tuple[Logger, str]:
    """Train the agent on CartPole environment."""
    env = gym.make("CartPole-v1")

    agent = Agent(config)
    memory_buffer = ReplayBuffer(capacity=config.replay_buffer_size)
    logger = Logger(log_dir="logs/cartpole")
    trainer = Trainer(agent, memory_buffer, config, logger, run_name=run_name)

    print("Starting CuMind training on CartPole...")
    print(f"Config: {config}")
    print(f"Checkpoints will be saved in: {trainer.checkpoint_dir}")

    trainer.run_training_loop(env, num_episodes=5000, train_frequency=10)
    print("Training completed!")

    env.close()
    return logger, trainer.checkpoint_dir


def get_latest_checkpoint(checkpoint_dir: str) -> str | None:
    """Gets the latest checkpoint file from a directory."""
    if not os.path.isdir(checkpoint_dir):
        print(f"Checkpoint directory not found: {checkpoint_dir}")
        return None

    checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith(".pkl")])
    if not checkpoints:
        print(f"No checkpoints found in {checkpoint_dir}.")
        return None

    return os.path.join(checkpoint_dir, checkpoints[-1])


def inference(config: Config, checkpoint_file: str) -> None:
    """Run inference with the trained agent from a checkpoint file."""
    print("\nStarting inference...")

    if not os.path.isfile(checkpoint_file):
        print(f"Checkpoint file not found: {checkpoint_file}")
        return

    print(f"Loading agent from: {checkpoint_file}")

    # Load agent from checkpoint
    inference_agent = Agent(config)
    state = load_checkpoint(checkpoint_file)
    inference_agent.load_state(state)

    # Run inference episodes
    inference_env = gym.make("CartPole-v1", render_mode="human")
    for episode in range(5):
        obs, _ = inference_env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action, _ = inference_agent.select_action(obs, training=False)
            obs, reward, terminated, truncated, _ = inference_env.step(action)
            total_reward += float(reward)
            done = terminated or truncated
        print(f"Inference Episode {episode + 1}: Total Reward = {total_reward}")

    inference_env.close()


def main() -> None:
    """Main function orchestrating training and inference."""
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

    # logger, checkpoint_dir = train(config)
    # logger.close()

    # latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
    latest_checkpoint = "/home/dreamland/workspace/cumind/checkpoints/cartpole_episode_04950.pkl"
    if latest_checkpoint:
        inference(config, latest_checkpoint)


if __name__ == "__main__":
    main()
