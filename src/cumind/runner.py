"""High-level training and inference runners."""

import os

import gymnasium as gym

from .agent.agent import Agent
from .agent.trainer import Trainer
from .utils.checkpoint import load_checkpoint
from .utils.config import cfg
from .utils.logger import log


def train() -> str:
    """Train the agent on a given environment."""
    env = gym.make(cfg.env.name)

    agent = Agent()
    memory_buffer = cfg.memory()
    trainer = Trainer(agent, memory_buffer)

    log.info(f"Checkpoints will be saved in: {trainer.checkpoint_dir}")
    log.info("Starting training...")
    trainer.run_training_loop(env)
    log.info("Training completed!")

    env.close()  # type: ignore
    return trainer.checkpoint_dir


def inference(checkpoint_file: str) -> None:
    """Run inference with a trained agent from a checkpoint."""
    log.info("\nStarting inference...")

    if not os.path.isfile(checkpoint_file):
        log.error(f"Checkpoint file not found: {checkpoint_file}")
        return

    log.info(f"Loading agent from: {checkpoint_file}")

    inference_agent = Agent()
    state = load_checkpoint(checkpoint_file)
    inference_agent.load_state(state)

    env = gym.make(cfg.env.name, render_mode="human")
    for episode in range(500):
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
