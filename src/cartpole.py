"""Basic training example for CuMind with CartPole."""

import jax.profiler

from cumind.agent.runner import inference, train
from cumind.utils.config import cfg
from cumind.utils.logger import log


def main() -> None:
    """Main function for running the CartPole example."""
    timestamp, checkpoint_dir = cfg.load("test.json")
    print(f"Run UUID: {timestamp}")

    train()
    log.info(f"Training completed in {log.elapsed()}.")

    latest_ckpt = f"{checkpoint_dir}/episode_{cfg.training.num_episodes:05d}.pkl"

    log.open()
    inference(latest_ckpt, 500)


if __name__ == "__main__":
    with jax.profiler.trace("/tmp/profile-data"):
        main()
    jax.profiler.save_device_memory_profile("memory.prof")
    log.shutdown()
