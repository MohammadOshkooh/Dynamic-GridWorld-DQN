"""Moving-goal variant of the existing partially observable NavigationGridWorld.

This module is deliberately separate from ``environment.py``. It reuses the
baseline map, agent, patrols, 24-dimensional observation and reward settings,
while adding one change: the visible goal performs a one-cell random walk every
``goal_move_interval`` environment steps.

The goal is kept off fixed walls and all patrol-route cells. This preserves the
baseline patrol behavior and prevents the goal from overlapping an enemy.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from navigation_rl.environment import ACTIONS, NavigationGridWorld


DEFAULT_GOAL_MOVE_INTERVAL = 3


class MovingGoalNavigationGridWorld(NavigationGridWorld):
    """Baseline environment with a visible goal that moves during each episode.

    Behavior and timing:
      * The goal is sampled randomly at reset, just like the baseline.
      * The goal remains still for the first ``goal_move_interval - 1`` steps.
      * After the agent's action and patrol update on each interval boundary,
        the goal randomly moves to one free cardinal neighbor.
      * Goal movement cannot enter a wall, a patrol-route cell, a currently
        occupied patrol cell, or the agent's current cell.
      * If no legal neighboring cell exists, the goal waits in place.
      * Reaching the currently visible goal ends the episode immediately,
        before that step's scheduled goal movement.

    The goal's current row and column are already part of the base observation,
    so the observation remains 24-dimensional and the DQN architecture does not
    need to change.
    """

    def __init__(
        self,
        *args: Any,
        goal_move_interval: int = DEFAULT_GOAL_MOVE_INTERVAL,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.goal_move_interval = int(goal_move_interval)
        if self.goal_move_interval <= 0:
            raise ValueError("goal_move_interval must be a positive integer")

        # Keep the moving goal away from every cell used by a patrol route.
        # That means the original patrols do not need modified collision logic.
        self._safe_goal_cells = frozenset(
            self._reachable_from_start - self.static_walls - self._path_cells
        )
        if not self._safe_goal_cells:
            raise ValueError("No safe cells are available for the moving goal")
        self.goal_moves = 0

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        observation, base_info = super().reset(seed=seed, options=options)
        self.goal_moves = 0
        info = dict(base_info)
        info.update(
            {
                "goal_moved": False,
                "goal_moves": self.goal_moves,
                "goal_move_interval": self.goal_move_interval,
            }
        )
        self._last_info = info
        return observation, info

    def _advance_goal(self) -> bool:
        """Move the goal by one legal cardinal cell; return whether it moved."""
        occupied = set(self.moving_obstacle_positions)
        occupied.add(self.agent_pos)

        candidates: list[tuple[int, int]] = []
        row, col = self.goal_pos
        for dr, dc in ACTIONS.values():
            candidate = (row + dr, col + dc)
            if candidate in self._safe_goal_cells and candidate not in occupied:
                candidates.append(candidate)

        if not candidates:
            return False

        # The environment RNG is seeded by reset(), making goal motion
        # reproducible for a fixed seed and action sequence.
        selected = candidates[int(self.rng.integers(len(candidates)))]
        if selected == self.goal_pos:
            return False

        self.goal_pos = selected
        self._distance_to_goal = self._build_goal_distances(self.goal_pos)
        self.goal_moves += 1
        return True

    def step(self, action: int):
        """Advance the agent, patrols and (when scheduled) the moving goal."""
        if not isinstance(action, (int, np.integer)) or int(action) not in ACTIONS:
            raise ValueError(f"action must be one of {sorted(ACTIONS)}")

        action = int(action)
        self.steps += 1
        old_position = self.agent_pos
        old_goal_position = self.goal_pos
        old_potential = self._potential(old_position)

        dr, dc = ACTIONS[action]
        candidate = (old_position[0] + dr, old_position[1] + dc)

        blocked_reason = None
        if not self._in_bounds(candidate):
            blocked_reason = "boundary"
        elif candidate in self.static_walls:
            blocked_reason = "static_wall"
        elif candidate in self.moving_obstacle_positions:
            blocked_reason = "moving_obstacle"

        if blocked_reason is None:
            self.agent_pos = candidate

        # Capture the goal before it moves: entering the currently visible
        # target cell is an immediate success, as in the baseline environment.
        terminated = self.agent_pos == self.goal_pos
        goal_moved = False

        if not terminated:
            # Keep the baseline update order: patrols move after the agent.
            if self.steps % self.move_interval == 0:
                self._advance_patrols()

            # The goal moves after the agent and patrols, on its own cadence.
            if self.steps % self.goal_move_interval == 0:
                goal_moved = self._advance_goal()

        truncated = self.steps >= self.max_steps and not terminated

        # If the goal moved, its new position is used for the next state's
        # potential. The old potential was computed against the old goal.
        new_potential = 0.0 if terminated else self._potential(self.agent_pos)
        base_reward = self.goal_reward if terminated else self.step_penalty
        if blocked_reason is not None:
            base_reward += self.blocked_penalty

        # Same potential-based shaping formula and gamma as the baseline.
        shaping_reward = self.gamma * new_potential - old_potential
        reward = float(base_reward + shaping_reward)

        reason = "goal_reached" if terminated else ("time_limit" if truncated else None)
        info = {
            "reason": reason,
            "success": bool(terminated),
            "blocked": blocked_reason is not None,
            "blocked_reason": blocked_reason,
            "agent_position": self.agent_pos,
            "goal_position": self.goal_pos,
            "previous_goal_position": old_goal_position,
            "goal_moved": bool(goal_moved),
            "goal_moves": self.goal_moves,
            "goal_move_interval": self.goal_move_interval,
            "moving_obstacle_positions": list(self.moving_obstacle_positions),
            "steps": self.steps,
            "base_reward": float(base_reward),
            "shaping_reward": float(shaping_reward),
        }
        self._last_info = info
        return self._get_observation(), reward, terminated, truncated, info
