"""Train and evaluate DQN in the moving-goal / stochastic-enemy environment.

Example (Windows PowerShell, from the project root):
    python scripts/train_stochastic_enemy_dqn.py --episodes 1000 --max-steps 500
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
from stochastic_enemy_dqn.environment import (  # noqa: E402
    DEFAULT_ENEMY_PAUSE_DURATION,
    DEFAULT_ENEMY_PAUSE_PROBABILITY,
    DEFAULT_ENEMY_SPEED_INTERVALS,
    StochasticEnemyNavigationGridWorld,
)
from stochastic_enemy_dqn.evaluation import evaluate_stochastic_enemy_policy  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--goal-move-interval", type=int, default=3)
    parser.add_argument("--pause-probability", type=float, default=DEFAULT_ENEMY_PAUSE_PROBABILITY)
    parser.add_argument("--output", type=str, default="results_stochastic_enemies")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    args = parser.parse_args()

    if args.episodes <= 0 or args.eval_episodes <= 0 or args.max_steps <= 0:
        parser.error("episodes, eval-episodes and max-steps must be positive")
    if args.goal_move_interval <= 0:
        parser.error("goal-move-interval must be positive")
    if not 0 <= args.pause_probability <= 1:
        parser.error("pause-probability must be in [0, 1]")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    dqn_config = DQNConfig(device=args.device)
    training_config = TrainingConfig(
        episodes=args.episodes,
        seed=args.seed,
        evaluation_episodes=args.eval_episodes,
        evaluation_seed=50_000,
        max_steps=args.max_steps,
        log_interval=max(1, min(20, args.episodes)),
        checkpoint_interval=0,
        output_dir=str(output_dir),
        show_progress=True,
    )
    env = StochasticEnemyNavigationGridWorld(
        gamma=dqn_config.gamma,
        max_steps=args.max_steps,
        goal_move_interval=args.goal_move_interval,
        enemy_speed_intervals=DEFAULT_ENEMY_SPEED_INTERVALS,
        pause_probability=args.pause_probability,
        pause_duration=DEFAULT_ENEMY_PAUSE_DURATION,
    )
    agent = DQNAgent(dqn_config, seed=args.seed)

    print("\nStochastic-enemy DQN experiment")
    print("Episodes:", args.episodes)
    print("Max steps:", args.max_steps)
    print("Goal move interval:", args.goal_move_interval)
    print("Enemy speed intervals:", DEFAULT_ENEMY_SPEED_INTERVALS)
    print("Pause probability:", args.pause_probability)
    print("Pause duration:", DEFAULT_ENEMY_PAUSE_DURATION)
    print("Output:", output_dir.resolve())

    history = train_agent(agent, env, training_config)
    metrics = evaluate_stochastic_enemy_policy(
        agent,
        num_episodes=args.eval_episodes,
        seed=training_config.evaluation_seed,
        max_steps=args.max_steps,
        goal_move_interval=args.goal_move_interval,
        pause_probability=args.pause_probability,
    )
    pd.DataFrame([metrics]).to_csv(output_dir / "evaluation_stochastic_enemies.csv", index=False)
    agent.save(
        output_dir / "dqn_stochastic_enemies_final.pt",
        extra={
            "scenario": "moving_goal_stochastic_enemies",
            "episodes": args.episodes,
            "goal_move_interval": args.goal_move_interval,
            "enemy_speed_intervals": list(DEFAULT_ENEMY_SPEED_INTERVALS),
            "pause_probability": args.pause_probability,
            "pause_duration": list(DEFAULT_ENEMY_PAUSE_DURATION),
        },
    )
    figure = plot_training_history(history, window=100)
    figure.savefig(output_dir / "training_curves_stochastic_enemies.png",
                   dpi=160, bbox_inches="tight")
    print("\nGreedy evaluation:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    print("Training history:", (output_dir / "training_history.csv").resolve())
    print("Checkpoint:", (output_dir / "dqn_stochastic_enemies_final.pt").resolve())


if __name__ == "__main__":
    main()
