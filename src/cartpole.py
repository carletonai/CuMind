"""Basic training example for CuMind with CartPole."""

from cumind.runner import inference, train
from cumind.utils.config import cfg
from cumind.utils.logger import log


def main() -> None:
    """Main function for running the CartPole example."""
    # Loads whatever configuration in config.py
    cfg.load("configuration.json")
    log.set_level("DEBUG")
    # log.set_file(cfg.logging.level)
    log.open()
    ckpt = train()
    inference(ckpt)


if __name__ == "__main__":
    main()
