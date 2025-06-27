"""High-level training and inference runners."""

import os

import gymnasium as gym

from .agent.agent import Agent
from .agent.trainer import Trainer
from .config import Config
from .data.memory import MemoryBuffer
from .utils.checkpoint import load_checkpoint
from .utils.logger import log


def train(config: Config) -> str:
    """Train the agent on a given environment."""
    env = gym.make(config.env_name)

    agent = Agent(config)
    memory_buffer = MemoryBuffer(capacity=config.memory_capacity)
    trainer = Trainer(agent, memory_buffer, config)

    log.info(f"Checkpoints will be saved in: {trainer.checkpoint_dir}")
    log.info("Starting training...")
    trainer.run_training_loop(env)
    log.info("Training completed!")

    env.close()  # type: ignore
    return trainer.checkpoint_dir


def inference(config: Config, checkpoint_file: str) -> None:
    """Run inference with a trained agent from a checkpoint."""
    log.info("\nStarting inference...")

    if not os.path.isfile(checkpoint_file):
        log.error(f"Checkpoint file not found: {checkpoint_file}")
        return

    log.info(f"Loading agent from: {checkpoint_file}")

    inference_agent = Agent(config)
    state = load_checkpoint(checkpoint_file)
    inference_agent.load_state(state)

    env = gym.make(config.env_name, render_mode="human")
    for episode in range(5):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action, _ = inference_agent.select_action(obs, training=False)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            done = terminated or truncated
        log.info(f"Inference Episode {episode + 1}: Total Reward = {total_reward}")

    env.close()  # type: ignore
