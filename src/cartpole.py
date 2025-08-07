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
    jax.profiler.start_trace("/tmp/profile-data", create_perfetto_link=True)
    timestamp, checkpoint_dir = cfg.load("test.json")
    print(f"Run UUID: {timestamp}")

    train()
    log.info(f"Training completed in {log.elapsed()}.")
    jax.profiler.stop_trace()  # type: ignore
    jax.profiler.save_device_memory_profile("memory.prof")
    log.shutdown()

    # Viewing Perfetto locally (for dev):
    # After program runs, follow the link printed in the terminal to view trace in your browser.
    # Or manually upload /tmp/profile-data to https://ui.perfetto.dev

    # Viewing Perfetto remotely (TPU, SSH):
    # If running on a remote server/TPU, set up an SSH tunnel before running main.py:
    # ssh -L 9001:127.0.0.1:9001 <your-user>@<remote-host>
    # Then, open the provided Perfetto link in your local browser.
    #
    # For Google Cloud TPU:
    # gcloud compute ssh <tpu-instance-name> -- -L 9001:127.0.0.1:9001
