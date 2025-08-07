"""High-level training and inference runners."""

import os

import gymnasium as gym

from cumind.agent.agent import Agent
from cumind.agent.trainer import Trainer
from cumind.utils.checkpoint import AgentState, find_latest_checkpoint_for_env, load_checkpoint
from cumind.utils.config import cfg
from cumind.utils.logger import log

from typing import Optional

def train(resume_from_latest: bool = False, checkpoint_path: Optional[str] = None) -> str:
    """Train the agent on a given environment.

    Args:
        resume_from_latest: If True, automatically resume from latest checkpoint for this env
        checkpoint_path: Specific checkpoint path to resume from
    """
    env = gym.make(id=cfg.env.name, max_episode_steps=cfg.env.max_episode_steps)

    agent = Agent()
    memory_buffer = cfg.memory()
    trainer = Trainer(agent, memory_buffer)

    # Determine checkpoint to resume from
    resume_checkpoint = None
    if checkpoint_path:
        if not os.path.isfile(checkpoint_path):
            raise RuntimeError(f"Checkpoint file not found: {checkpoint_path}")
        resume_checkpoint = checkpoint_path
    elif resume_from_latest:
        resume_checkpoint = find_latest_checkpoint_for_env(cfg.env.name)
        if resume_checkpoint:
            log.info(f"Found latest checkpoint for {cfg.env.name}: {resume_checkpoint}")

    trainer.train(env, resume_from_checkpoint=resume_checkpoint)

    env.close()  # type: ignore

    return trainer.checkpoint_dir


def inference(checkpoint_file: str, num_episodes: int) -> None:
    """Run inference with a trained agent from a checkpoint."""
    log.info("\nStarting inference.")

    if not os.path.isfile(checkpoint_file):
        raise RuntimeError(f"Checkpoint file not found: {checkpoint_file}")

    log.info(f"Loading agent from: {checkpoint_file}")

    inference_agent = Agent()
    agent_state: AgentState = load_checkpoint(checkpoint_file)["state"]
    inference_agent.load_state(agent_state)

    env = gym.make(id=cfg.env.name, max_episode_steps=cfg.env.max_episode_steps, render_mode="human")
    for episode in range(num_episodes):
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
