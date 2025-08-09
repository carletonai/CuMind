"""High-level training and inference runners."""

import os

import gymnasium as gym

from cumind.agent.agent import Agent
from cumind.agent.trainer import Trainer
from cumind.utils.checkpoint import load_checkpoint
from cumind.utils.config import cfg
from cumind.utils.logger import log


def train() -> None:
    env = None
    try:
        env = gym.make(id=cfg.env.name, max_episode_steps=cfg.env.max_episode_steps)
        Trainer(Agent(), cfg.memory()).train(env)
        log.info(f"Training completed in {log.elapsed()}.")
    except Exception as e:
        log.exception(f"Exception occurred: {e}")
    finally:
        if env is not None:
            env.close()  # type: ignore
        log.shutdown()


def inference(checkpoint_file: str, num_episodes: int) -> None:
    """Run inference with a trained agent from a checkpoint."""
    try:
        log.open()
        log.info("\nStarting inference.")

        if not os.path.isfile(checkpoint_file):
            raise RuntimeError(f"Checkpoint file not found: {checkpoint_file}")

        log.info(f"Loading agent from: {checkpoint_file}")

        agent = Agent().load_state(load_checkpoint(checkpoint_file))
        env = gym.make(id=cfg.env.name, max_episode_steps=cfg.env.max_episode_steps, render_mode="human")

        for episode in range(num_episodes):
            obs, _ = env.reset()
            done = False
            total_reward = 0.0
            while not done:
                action, _ = agent.select_action(obs, training=False)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)
                done = terminated or truncated
            log.info(f"Inference Episode {episode + 1}: Total Reward = {total_reward}")

        env.close()  # type: ignore
    except Exception as e:
        log.exception(f"Exception occurred: {e}")
    finally:
        log.shutdown()
