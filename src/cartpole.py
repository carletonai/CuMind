"""Basic training example for CuMind with CartPole."""

from cumind.config import Config
from cumind.runner import train


def main() -> None:
    """Main function for running the CartPole example."""
    # Loads whatever configuration in config.py
    config = Config()

    config.env_name = "CartPole-v1"
    config.action_space_size = 2
    config.observation_shape = (4,)
    config.num_episodes = 500

    train(config)


if __name__ == "__main__":
    main()
