"""Basic training example for CuMind with CartPole."""

from cumind.config import Config
from cumind.runner import train


def main() -> None:
    """Main function for running the CartPole example."""
    # Load default configuration
    config = Config()

    # Customize for CartPole
    config.env_name = "CartPole-v1"
    config.action_space_size = 2
    config.observation_shape = (4,)
    config.num_episodes = 500

    # Start training
    train(config)


if __name__ == "__main__":
    main()
