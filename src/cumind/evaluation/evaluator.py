"""evaluate trained models systematically"""

import time
from dataclasses import dataclass, asdict
from typing import List, Dict
import numpy as np
from tqdm import tqdm
import gym

from cumind.agent.agent import Agent
from cumind.utils.checkpoint import load_checkpoint
from .visualization import generate_plots_for_report


@dataclass
class EvalMetrics:
    """
    A dataclass to store metrics from an evaluation run.
    """

    checkpoint_path: str
    env_name: str
    num_episodes: int
    episode_rewards: List[float]
    episode_lengths: List[int]
    action_entropy: List[float]
    value_accuracy: List[float]
    mcts_depth: List[float]
    inference_time: List[float]
    success_rate: float = 0.0

    def summary(self) -> Dict[str, str]:
        """Returns a dictionary of formatted summary statistics."""
        mean_reward = np.mean(self.episode_rewards)
        std_reward = np.std(self.episode_rewards)

        return {
            "Mean Reward": f"{mean_reward:.2f} ± {std_reward:.2f}",
            "Success Rate": f"{self.success_rate * 100:.1f}%",
            "Mean Episode Length": f"{np.mean(self.episode_lengths):.1f}",
            "Mean MCTS Depth": (
                f"{np.mean(self.mcts_depth):.2f}" if self.mcts_depth else "N/A"
            ),
            "Mean Inference Time": f"{np.mean(self.inference_time) * 1000:.3f} ms/step",
        }


ComparisonReport = Dict[str, EvalMetrics]


class Evaluator:
    """
    Handles systematic evaluation of trained agents, checkpoint comparison,
    and report generation.
    """

    def evaluate_episodes(
        self, agent: Agent, env_name: str, num_episodes: int, seed: int = 42
    ) -> EvalMetrics:
        """
        Run evaluation episodes and collect metrics.

        Args:
            agent (Agent): The agent instance to evaluate.
            env_name (str): The name of the Gym environment.
            num_episodes (int): The number of episodes to run.
            seed (int): A seed for the environment to ensure reproducibility.

        Returns:
            EvalMetrics: An object containing the collected metrics.
        """
        print(f"Evaluating agent on {env_name} for {num_episodes} episodes...")
        env = gym.make(env_name)
        env.seed(seed)

        metrics = EvalMetrics(
            checkpoint_path=(
                agent.checkpoint_path if hasattr(agent, "checkpoint_path") else "N/A"
            ),
            env_name=env_name,
            num_episodes=num_episodes,
            episode_rewards=[],
            episode_lengths=[],
            action_entropy=[],
            value_accuracy=[],
            mcts_depth=[],
            inference_time=[],
        )

        num_successes = 0

        for _ in tqdm(range(num_episodes), desc="Running Episodes"):
            obs = env.reset()
            done = False
            episode_reward = 0
            episode_length = 0

            while not done:
                start_time = time.perf_counter()

                action, agent_metrics = agent.evaluate(obs)

                end_time = time.perf_counter()

                obs, reward, done, info = env.step(action)

                episode_reward += reward
                episode_length += 1

                metrics.inference_time.append(end_time - start_time)

                if agent_metrics:
                    metrics.mcts_depth.append(agent_metrics.get("mcts_depth", np.nan))
                    metrics.action_entropy.append(
                        agent_metrics.get("action_entropy", np.nan)
                    )

            metrics.episode_rewards.append(episode_reward)
            metrics.episode_lengths.append(episode_length)

            if (
                env_name.startswith("CartPole")
                and episode_length >= env.spec.max_episode_steps
            ):
                num_successes += 1

        metrics.success_rate = num_successes / num_episodes
        env.close()

        print("Evaluation complete")
        return metrics

    def compare_checkpoints(
        self, checkpoint_paths: List[str], env_name: str, num_episodes: int
    ) -> ComparisonReport:
        """
        Compare performance across multiple checkpoints.

        Args:
            checkpoint_paths (List[str]): List of paths to model checkpoints.
            env_name (str): The environment name to evaluate on.
            num_episodes (int): Number of episodes for each evaluation.

        Returns:
            ComparisonReport: A dictionary mapping checkpoint paths to their EvalMetrics.
        """
        print(f"Comparing {len(checkpoint_paths)} checkpoints...")
        report: ComparisonReport = {}

        for path in checkpoint_paths:
            print(f"\n--- Loading checkpoint: {path} ---")
            try:
                agent = load_checkpoint(path)
                agent.checkpoint_path = path
                metrics = self.evaluate_episodes(agent, env_name, num_episodes)
                report[path] = metrics
            except Exception as e:
                print(f"Failed to evaluate checkpoint {path}: {e}")

        return report

    def generate_report(self, metrics: EvalMetrics, output_path: str):
        """
        Generate an HTML report with plots and statistics.

        Args:
            metrics (EvalMetrics): The evaluation metrics to report.
            output_path (str): The path to save the HTML report.
        """
        print(f"Generating report at: {output_path}")

        summary_stats = metrics.summary()

        try:
            html_content = generate_plots_for_report(metrics, summary_stats)
            with open(output_path, "w") as f:
                f.write(html_content)
            print("Report saved successfully.")
        except Exception as e:
            print(f"Failed to generate report: {e}")
