"""Separate moving-goal experiment package built on the original navigation_rl package."""

from .environment import DEFAULT_GOAL_MOVE_INTERVAL, MovingGoalNavigationGridWorld
from .evaluation import evaluate_moving_goal_policy

__all__ = [
    "DEFAULT_GOAL_MOVE_INTERVAL",
    "MovingGoalNavigationGridWorld",
    "evaluate_moving_goal_policy",
]
