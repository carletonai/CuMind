"""Basic training example for CuMind with CartPole."""

from cumind.config import Config
from cumind.runner import train


def main() -> None:
    """Main function for running the CartPole example."""
    # Loads whatever configuration in config.py
    config = Config().from_json("configuration.json")
    train(config)


if __name__ == "__main__":
    main()
