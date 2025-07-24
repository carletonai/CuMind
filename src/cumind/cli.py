"""Command-line interface for CuMind."""

import argparse
import os
from typing import Optional

from cumind.agent.runner import inference, train
from cumind.utils.checkpoint import latest_checkpoints
from cumind.utils.config import cfg
from cumind.utils.logger import log


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="CuMind CLI")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    return parser.parse_args()


def select_checkpoint() -> Optional[str]:
    """Prompts the user to select a checkpoint and returns the path."""
    checkpoints = latest_checkpoints("checkpoints")
    if not checkpoints:
        log.info("No checkpoints found.")
        return None

    print("\nAvailable checkpoints:")
    options = []
    for env, runs in checkpoints.items():
        latest_run = runs[0]
        checkpoint_path, timestamp = latest_run
        print(f"  - {env}: Latest run from {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        options.append((checkpoint_path, f"{env} (latest)"))

    print("\nPlease choose an option:")
    print("  0: Start a new run")
    for i, (_, desc) in enumerate(options):
        print(f"  {i + 1}: Load from {desc}")
    print(f"  {len(options) + 1}: Load from a manual path")

    while True:
        try:
            choice = int(input(f"Enter your choice [0-{len(options) + 1}]: "))
            if 0 <= choice <= len(options) + 1:
                if choice == 0:
                    return None
                if choice == len(options) + 1:
                    return input("Enter the full path to the checkpoint file: ")
                return options[choice - 1][0]
            else:
                log.warning("Invalid choice, please try again.")
        except ValueError:
            log.warning("Invalid input, please enter a number.")


def main() -> None:
    """Main CLI entry point."""
    log.info("Welcome to CuMind!")

    args = parse_arguments()
    config_path = args.config
    if not os.path.exists(config_path):
        log.info(f"Default config not found at {config_path}. Creating default config there.")
        cfg.save(config_path)
    else:
        cfg.load(config_path)

    checkpoint_path = select_checkpoint()

    if checkpoint_path:
        log.info(f"Loading from checkpoint: {checkpoint_path}")
        inference(checkpoint_path)
    else:
        log.info("Starting a new run.")
        train()

    log.close()
