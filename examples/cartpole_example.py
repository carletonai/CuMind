"""Example usage of MuZero with Gymnasium environment."""

import gymnasium as gym

from muzero import MuZeroAgent, MuZeroConfig


def main():
    """Run a simple example with MuZero."""
    # Create environment
    env = gym.make("CartPole-v1")

    # Create MuZero configuration
    config = MuZeroConfig(
        action_space_size=env.action_space.n,
        observation_shape=(4,),  # CartPole observation space
        hidden_dim=64,
        num_simulations=25,  # Fewer simulations for faster demo
        batch_size=16,
    )

    # Create agent
    agent = MuZeroAgent(config)

    print("Running MuZero example with CartPole...")

    # Run a few episodes
    for episode in range(3):
        observation, _ = env.reset()
        total_reward = 0
        steps = 0

        while steps < 200:  # Limit episode length
            action = agent.select_action(observation, training=False)
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1

            if terminated or truncated:
                break

        print(f"Episode {episode + 1}: {steps} steps, reward: {total_reward}")

    env.close()
    print("Example completed!")


if __name__ == "__main__":
    main()
