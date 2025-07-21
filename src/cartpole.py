"""Basic training example for CuMind with CartPole."""

from cumind.agent.runner import inference, train
from cumind.utils.config import cfg
from cumind.utils.logger import log


def main() -> None:
    """Main function for running the CartPole example."""
    timestamp, checkpoint_dir = cfg.load("configuration.json")

    # Print directory information
    print(f"Logging directory: {cfg.logging.dir}/{timestamp}/training.log")
    print(f"Checkpoint directory: {checkpoint_dir}/")

    #train()
    log.info(f"Training completed in {log.elapsed()}.")

    # Print latest checkpoint path
    latest_ckpt = f"{checkpoint_dir}/episode_{cfg.training.num_episodes:05d}.pkl"
    latest_ckpt = "checkpoints/CartPole-v1/20250720_230602/episode_00700.pkl"
    print(f"Latest checkpoint: {latest_ckpt}")
    inference(latest_ckpt, 500)


if __name__ == "__main__":
    main()
    log.shutdown()
