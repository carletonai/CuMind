"""Basic training example for CuMind with CartPole."""

from cumind.config import Config
from cumind.runner import inference, train


def main() -> None:
    """Main function for running the CartPole example."""
    # Loads whatever configuration in config.py
    config = Config()
    ckpt = train(config)
    inference(config, ckpt)


if __name__ == "__main__":
    main()
