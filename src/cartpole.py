"""Basic training example for CuMind with CartPole."""

from cumind.agent.runner import inference, train
from cumind.utils.config import cfg
from cumind.utils.logger import log


def main() -> None:
    """Main function for running the CartPole example."""
    cfg.load("configuration.json")
    ckpt = train()
    log.info(f"Training completed in {log.elapsed()}.")
    inference(ckpt)


if __name__ == "__main__":
    main()
    log.shutdown()
