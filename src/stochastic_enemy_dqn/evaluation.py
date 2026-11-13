"""Evaluation helpers for the stochastic-enemy DQN experiment."""

from __future__ import annotations

from typing import Any

import numpy as np

from navigation_rl.dqn import DQNAgent
from .environment import (
    DEFAULT_ENEMY_PAUSE_DURATION,
    DEFAULT_ENEMY_PAUSE_PROBABILITY,
    DEFAULT_ENEMY_SPEED_INTERVALS,
    StochasticEnemyNavigationGridWorld,
)


def evaluate_stochastic_enemy_policy(
    agent: DQNAgent,
    num_episodes: int = 200,
    seed: int = 50_000,
    max_steps: int = 300,
    goal_move_interval: int = 3,
    enemy_speed_intervals: tuple[int, ...] = DEFAULT_ENEMY_SPEED_INTERVALS,
    pause_probability: float = DEFAULT_ENEMY_PAUSE_PROBABILITY,
    pause_duration: tuple[int, int] = DEFAULT_ENEMY_PAUSE_DURATION,
) -> dict[str, float]:
    """Greedily evaluate a policy on fresh, seeded stochastic-enemy episodes."""
    if num_episodes <= 0:
        raise ValueError("num_episodes must be positive")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    env = StochasticEnemyNavigationGridWorld(
        gamma=agent.config.gamma,
        max_steps=max_steps,
        goal_move_interval=goal_move_interval,
        enemy_speed_intervals=enemy_speed_intervals,
        pause_probability=pause_probability,
        pause_duration=pause_duration,
    )

    returns: list[float] = []
    lengths: list[int] = []
    successes: list[float] = []
    blocked_rates: list[float] = []
    goal_move_counts: list[int] = []
    enemy_move_counts: list[int] = []
    enemy_pause_counts: list[int] = []

    for episode in range(num_episodes):
        observation, _ = env.reset(seed=seed + episode)
        total_reward = 0.0
        blocked = 0
        moved = 0
        paused = 0
        terminated = truncated = False
        info: dict[str, Any] = {}

        while not (terminated or truncated):
            action = agent.choose_action(observation, explore=False)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            blocked += int(info["blocked"])
            moved += int(info["enemies_moved_count"])
            paused += int(info["enemies_paused_count"])

        length = int(info.get("steps", 0))
        returns.append(total_reward)
        lengths.append(length)
        successes.append(float(info.get("success", False)))
        blocked_rates.append(blocked / max(length, 1))
        goal_move_counts.append(int(info.get("goal_moves", 0)))
        enemy_move_counts.append(moved)
        enemy_pause_counts.append(paused)

    return {
        "success_rate": float(np.mean(successes)),
        "average_return": float(np.mean(returns)),
        "average_steps": float(np.mean(lengths)),
        "blocked_action_rate": float(np.mean(blocked_rates)),
        "average_goal_moves": float(np.mean(goal_move_counts)),
        "average_enemy_moves_per_episode": float(np.mean(enemy_move_counts)),
        "average_enemy_paused_steps_per_episode": float(np.mean(enemy_pause_counts)),
    }
