"""Evaluation helpers for the moving-goal DQN experiment.

Training itself reuses ``navigation_rl.dqn.train_agent`` unchanged. This keeps
all original DQN hyperparameters, replay-buffer behavior and checkpoint format.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from navigation_rl.dqn import DQNAgent
from .environment import MovingGoalNavigationGridWorld


def evaluate_moving_goal_policy(
    agent: DQNAgent,
    num_episodes: int = 200,
    seed: int = 50_000,
    max_steps: int = 300,
    goal_move_interval: int = 3,
) -> dict[str, float]:
    """Evaluate a DQN greedily on seeded moving-goal episodes.

    No exploration, gradient updates or replay-buffer writes occur here.
    Returned metrics are averages over episodes, not over individual steps.
    """
    if num_episodes <= 0:
        raise ValueError("num_episodes must be positive")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if goal_move_interval <= 0:
        raise ValueError("goal_move_interval must be positive")

    env = MovingGoalNavigationGridWorld(
        gamma=agent.config.gamma,
        max_steps=max_steps,
        goal_move_interval=goal_move_interval,
    )
    returns: list[float] = []
    lengths: list[int] = []
    successes: list[float] = []
    blocked_rates: list[float] = []
    goal_move_counts: list[int] = []

    for episode in range(num_episodes):
        observation, _ = env.reset(seed=seed + episode)
        total_reward = 0.0
        blocked = 0
        terminated = truncated = False
        info: dict[str, Any] = {}

        while not (terminated or truncated):
            action = agent.choose_action(observation, explore=False)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            blocked += int(info["blocked"])

        length = int(info.get("steps", 0))
        returns.append(total_reward)
        lengths.append(length)
        successes.append(float(info.get("success", False)))
        blocked_rates.append(blocked / max(length, 1))
        goal_move_counts.append(int(info.get("goal_moves", 0)))

    return {
        "success_rate": float(np.mean(successes)),
        "average_return": float(np.mean(returns)),
        "average_steps": float(np.mean(lengths)),
        "blocked_action_rate": float(np.mean(blocked_rates)),
        "average_goal_moves": float(np.mean(goal_move_counts)),
    }
