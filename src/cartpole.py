"""Basic training example for CuMind with CartPole."""

import os
from datetime import datetime

import gymnasium as gym

from cumind.agent.agent import Agent
from cumind.agent.trainer import Trainer
from cumind.config import Config
from cumind.data.memory import MemoryBuffer
from cumind.utils.checkpoint import find_latest_checkpoint_in_dir, load_checkpoint
from cumind.utils.logger import log


def train(config: Config) -> str:
    """Train the agent on CartPole environment."""
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
    """Run inference with the trained agent from a checkpoint file."""
    log.info("\nStarting inference...")

    if not os.path.isfile(checkpoint_file):
        log.error(f"Checkpoint file not found: {checkpoint_file}")
        return

    log.info(f"Loading agent from: {checkpoint_file}")

    # Load agent from checkpoint
    inference_agent = Agent(config)
    state = load_checkpoint(checkpoint_file)
    inference_agent.load_state(state)

    # Run inference episodes
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


def main() -> None:
    """Main function orchestrating training and inference."""
    config = Config.from_json("configuration.json")
    log.info(f"Using configuration: {config}")
    config.validate()

    should_train = True
    should_infer = False

    checkpoint_dir = ""
    if should_train:
        checkpoint_dir = train(config)

    if should_infer:
        manual_checkpoint = ""
        # Or use the one from the recent training run
        latest_checkpoint = find_latest_checkpoint_in_dir(checkpoint_dir) if checkpoint_dir else None

        if latest_checkpoint:
            inference(config, latest_checkpoint)
        elif manual_checkpoint and os.path.isfile(manual_checkpoint):
            log.info(f"Using manual checkpoint: {manual_checkpoint}")
            inference(config, manual_checkpoint)
        else:
            log.warning("No checkpoint found for inference.")

    log.close()


if __name__ == "__main__":
    main()
