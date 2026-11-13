"""Train and evaluate the existing DQN on the moving-goal environment.

Run from the project root, for example:
    python scripts/train_moving_goal_dqn.py --episodes 200 --max-steps 300

Optionally evaluate the old fixed-goal checkpoint on the moving-goal task first:
    python scripts/train_moving_goal_dqn.py --baseline-checkpoint results/dqn_final.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from navigation_rl.dqn import (  # noqa: E402
    DQNAgent,
    DQNConfig,
    TrainingConfig,
    plot_training_history,
    train_agent,
)
from moving_goal_dqn.evaluation import evaluate_moving_goal_policy  # noqa: E402
from moving_goal_dqn.environment import (  # noqa: E402
    DEFAULT_GOAL_MOVE_INTERVAL,
    MovingGoalNavigationGridWorld,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument(
        "--goal-move-interval", type=int, default=DEFAULT_GOAL_MOVE_INTERVAL,
        help="Move the goal once every N environment steps (default: 3).",
    )
    parser.add_argument("--output", type=str, default="results_moving_goal")
    parser.add_argument("--baseline-checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    args = parser.parse_args()

    if args.episodes <= 0 or args.eval_episodes <= 0 or args.max_steps <= 0:
        parser.error("episodes, eval-episodes and max-steps must be positive")
    if args.goal_move_interval <= 0:
        parser.error("goal-move-interval must be positive")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    dqn_config = DQNConfig(device=args.device)
    evaluation_rows: list[dict[str, float | str]] = []

    # Optional zero-shot comparison: the unchanged fixed-goal DQN is evaluated
    # on moving goals before a new model is trained.
    if args.baseline_checkpoint:
        baseline_path = Path(args.baseline_checkpoint)
        if not baseline_path.is_absolute():
            baseline_path = PROJECT_ROOT / baseline_path
        if not baseline_path.is_file():
            parser.error(f"Baseline checkpoint not found: {baseline_path}")

        baseline_agent = DQNAgent.load(
            baseline_path,
            map_location=args.device or "cpu",
        )
        baseline_metrics = evaluate_moving_goal_policy(
            baseline_agent,
            num_episodes=args.eval_episodes,
            seed=50_000,
            max_steps=args.max_steps,
            goal_move_interval=args.goal_move_interval,
        )
        evaluation_rows.append({"model": "fixed_goal_dqn_zero_shot", **baseline_metrics})
        print("\nExisting fixed-goal DQN evaluated on moving goals (no retraining):")
        for key, value in baseline_metrics.items():
            print(f"{key}: {value:.4f}")

    training_config = TrainingConfig(
        episodes=args.episodes,
        seed=args.seed,
        evaluation_episodes=args.eval_episodes,
        evaluation_seed=50_000,
        max_steps=args.max_steps,
        log_interval=max(1, min(100, args.episodes)),
        checkpoint_interval=0,
        output_dir=str(output_dir),
        show_progress=True,
    )

    agent = DQNAgent(dqn_config, seed=args.seed)
    env = MovingGoalNavigationGridWorld(
        gamma=dqn_config.gamma,
        max_steps=args.max_steps,
        goal_move_interval=args.goal_move_interval,
    )

    print("\nTraining a fresh DQN on the moving-goal environment")
    print(f"Episodes: {args.episodes}")
    print(f"Max steps per episode: {args.max_steps}")
    print(f"Goal moves every: {args.goal_move_interval} steps")
    print(f"Output directory: {output_dir.resolve()}")

    history = train_agent(agent, env, training_config)

    trained_metrics = evaluate_moving_goal_policy(
        agent,
        num_episodes=args.eval_episodes,
        seed=training_config.evaluation_seed,
        max_steps=args.max_steps,
        goal_move_interval=args.goal_move_interval,
    )
    evaluation_rows.append({"model": "moving_goal_dqn_trained", **trained_metrics})

    evaluation_path = output_dir / "evaluation_moving_goal.csv"
    pd.DataFrame(evaluation_rows).to_csv(evaluation_path, index=False)

    # Save an explicitly named copy with scenario metadata; the generic
    # dqn_final.pt is also produced by the original train_agent function.
    agent.save(
        output_dir / "dqn_moving_goal_final.pt",
        extra={
            "scenario": "moving_goal_random_walk",
            "goal_move_interval": args.goal_move_interval,
            "max_steps": args.max_steps,
            "episodes": args.episodes,
        },
    )

    try:
        figure = plot_training_history(history)
        figure.savefig(output_dir / "training_curves_moving_goal.png", dpi=160, bbox_inches="tight")
    except Exception as exc:  # Plotting must not invalidate the completed training run.
        print(f"Training plot could not be saved: {exc}")

    print("\nGreedy evaluation of the newly trained moving-goal DQN:")
    for key, value in trained_metrics.items():
        print(f"{key}: {value:.4f}")
    print(f"\nTraining history: {(output_dir / 'training_history.csv').resolve()}")
    print(f"Evaluation table: {evaluation_path.resolve()}")
    print(f"Checkpoint: {(output_dir / 'dqn_moving_goal_final.pt').resolve()}")


if __name__ == "__main__":
    main()
