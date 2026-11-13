"""Command-line entry point for a single-seed DQN training run."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from navigation_rl.dqn import DQNAgent, DQNConfig, TrainingConfig, train_agent, evaluate_policy, plot_training_history
from navigation_rl.environment import NavigationGridWorld


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--output", type=str, default="results")
    parser.add_argument("--device", type=str, default=None, choices=[None, "cpu", "cuda"])
    args = parser.parse_args()

    dqn_config = DQNConfig(device=args.device)
    training_config = TrainingConfig(
        episodes=args.episodes,
        seed=args.seed,
        evaluation_episodes=args.eval_episodes,
        output_dir=args.output,
    )
    agent = DQNAgent(dqn_config, seed=args.seed)
    env = NavigationGridWorld(gamma=dqn_config.gamma)
    history = train_agent(agent, env, training_config)
    metrics = evaluate_policy(
        agent,
        num_episodes=training_config.evaluation_episodes,
        seed=training_config.evaluation_seed,
        max_steps=training_config.max_steps,
    )
    print("\nGreedy evaluation")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    print(f"\nSaved checkpoint and history to: {Path(args.output).resolve()}")

    try:
        fig = plot_training_history(history)
        plot_path = Path(args.output) / "training_curves.png"
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, dpi=160, bbox_inches="tight")
        print(f"Saved plot to: {plot_path.resolve()}")
    except Exception as exc:
        print(f"Plot saving skipped: {exc}")


if __name__ == "__main__":
    main()
