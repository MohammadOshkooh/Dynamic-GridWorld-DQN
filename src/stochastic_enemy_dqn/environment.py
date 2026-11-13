"""Moving-goal GridWorld with independent stochastic enemies.

This experiment extends the moving-goal task from ``moving_goal_dqn``. The
number and layout of static walls, agent controls, local sensing, reward
function, and DQN observation size are kept compatible with the earlier
notebooks. Deterministic patrol routes are replaced by independent random
walkers with heterogeneous movement intervals and explicit pause durations.
"""

from __future__ import annotations

from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from navigation_rl.environment import ACTIONS, OBSERVATION_DIM
from moving_goal_dqn.environment import MovingGoalNavigationGridWorld

DEFAULT_ENEMY_SPEED_INTERVALS = (1, 2, 3, 4, 5)
DEFAULT_ENEMY_PAUSE_PROBABILITY = 0.20
DEFAULT_ENEMY_PAUSE_DURATION = (1, 4)
DEFAULT_ENEMY_SPAWN_MIN_DISTANCE = 4


class StochasticEnemyNavigationGridWorld(MovingGoalNavigationGridWorld):
    """Moving-goal environment populated by heterogeneous random-walk enemies.

    Each enemy receives a distinct movement interval. An interval of 1 means
    that the enemy may attempt one move every environment step; an interval
    of 5 means one movement opportunity every five steps. At each opportunity
    the enemy either chooses a uniformly random legal cardinal neighbor or,
    with ``pause_probability``, pauses for a random number of steps.

    Enemies never overlap one another, the agent, static walls, or the current
    goal. If no legal neighbor exists, an enemy waits until its next movement
    opportunity. Enemy positions are partially observable through the same
    four-cardinal-cell sensor used in versions 02 and 03.

    The returned observation remains 24-dimensional. The last two entries,
    previously used for deterministic patrol phase, are now reserved zeros:
    stochastic walkers have no shared deterministic phase to expose.
    """

    def __init__(
        self,
        *args: Any,
        enemy_speed_intervals: Iterable[int] = DEFAULT_ENEMY_SPEED_INTERVALS,
        pause_probability: float = DEFAULT_ENEMY_PAUSE_PROBABILITY,
        pause_duration: tuple[int, int] = DEFAULT_ENEMY_PAUSE_DURATION,
        spawn_min_distance: int = DEFAULT_ENEMY_SPAWN_MIN_DISTANCE,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        raw_intervals = tuple(enemy_speed_intervals)
        try:
            intervals = tuple(int(value) for value in raw_intervals)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("enemy_speed_intervals must contain integers") from exc
        if any(isinstance(value, (bool, np.bool_)) or int(value) != value
               for value in raw_intervals):
            raise ValueError("enemy_speed_intervals must contain integers")
        if not intervals:
            raise ValueError("enemy_speed_intervals must contain at least one value")
        if any(value <= 0 for value in intervals):
            raise ValueError("Every enemy speed interval must be a positive integer")
        if len(set(intervals)) != len(intervals):
            raise ValueError("Enemy speed intervals must be distinct to ensure heterogeneous speeds")
        if not 0.0 <= float(pause_probability) <= 1.0:
            raise ValueError("pause_probability must be in [0, 1]")
        try:
            raw_pause_duration = tuple(pause_duration)
            if len(raw_pause_duration) != 2:
                raise ValueError
            pause_min, pause_max = (int(value) for value in raw_pause_duration)
            if any(isinstance(value, (bool, np.bool_)) or int(value) != value
                   for value in raw_pause_duration):
                raise ValueError
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("pause_duration must be an integer (minimum, maximum) pair") from exc
        if pause_min <= 0 or pause_max < pause_min:
            raise ValueError("pause_duration must be a positive (minimum, maximum) pair")
        if (
            isinstance(spawn_min_distance, (bool, np.bool_))
            or int(spawn_min_distance) != spawn_min_distance
            or int(spawn_min_distance) < 0
        ):
            raise ValueError("spawn_min_distance must be a non-negative integer")

        self.default_enemy_speed_intervals = intervals
        self.pause_probability = float(pause_probability)
        self.pause_duration = (pause_min, pause_max)
        self.spawn_min_distance = int(spawn_min_distance)
        self.enemy_count = len(intervals)

        # Random walkers can traverse any reachable non-wall cell. The old
        # patrol-route cells are no longer special or reserved for the goal.
        self._safe_goal_cells = frozenset(self._reachable_from_start - self.static_walls)
        self._goal_candidates = sorted(self._safe_goal_cells - {self.start_position})
        if len(self._safe_goal_cells) <= self.enemy_count + 1:
            raise ValueError("The map does not contain enough free cells for the enemies and goal")

        self.enemy_speed_intervals = list(intervals)
        self.enemy_pause_remaining = [0] * self.enemy_count
        self.enemy_next_move_steps = [1] * self.enemy_count
        self.enemy_moved_this_step = [False] * self.enemy_count
        self.enemy_paused_this_step = [False] * self.enemy_count

    @property
    def enemy_positions(self) -> list[tuple[int, int]]:
        """Current enemy positions (a semantic alias for moving obstacles)."""
        return list(self.moving_obstacle_positions)

    def _sample_enemy_positions(
        self,
        excluded: set[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        # Sort before sampling so a seeded reset is reproducible across
        # separate Python processes (set iteration order is not a contract).
        candidates = sorted(
            cell for cell in self._safe_goal_cells
            if cell not in excluded
        )
        distant = [
            cell for cell in candidates
            if self._manhattan(cell, self.agent_pos) >= self.spawn_min_distance
        ]
        if len(distant) >= self.enemy_count:
            candidates = distant
        if len(candidates) < self.enemy_count:
            raise RuntimeError("Not enough free cells to place all enemies without overlap")

        selected_indices = self.rng.choice(len(candidates), size=self.enemy_count, replace=False)
        return [tuple(candidates[int(index)]) for index in selected_indices]

    def _reset_enemy_dynamics(self) -> None:
        # Shuffle the distinct speed profile so enemy identity does not imply
        # a fixed speed across episodes, while preserving heterogeneity.
        self.enemy_speed_intervals = list(self.default_enemy_speed_intervals)
        self.rng.shuffle(self.enemy_speed_intervals)
        self.enemy_pause_remaining = [0] * self.enemy_count
        self.enemy_next_move_steps = [
            int(self.rng.integers(1, interval + 1))
            for interval in self.enemy_speed_intervals
        ]
        self.enemy_moved_this_step = [False] * self.enemy_count
        self.enemy_paused_this_step = [False] * self.enemy_count

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        # The parent handles seed initialization and moving-goal setup. The
        # free-cell candidate list has already been widened in __init__.
        _, base_info = super().reset(seed=seed, options=options)

        excluded = {self.agent_pos, self.goal_pos}
        self.moving_obstacle_positions = self._sample_enemy_positions(excluded)
        self._reset_enemy_dynamics()

        info = dict(base_info)
        info.update(self._enemy_info())
        info["moving_obstacle_positions"] = self.enemy_positions
        info["enemy_count"] = self.enemy_count
        info["pause_probability"] = self.pause_probability
        info["pause_duration"] = self.pause_duration
        observation = self._get_observation()
        self._last_info = info
        return observation, info

    def _enemy_info(self) -> dict[str, Any]:
        return {
            "enemy_positions": self.enemy_positions,
            "enemy_speed_intervals": [int(value) for value in self.enemy_speed_intervals],
            "enemy_pause_remaining": [int(value) for value in self.enemy_pause_remaining],
            "enemy_next_move_steps": [int(value) for value in self.enemy_next_move_steps],
            "enemy_moved_this_step": [bool(value) for value in self.enemy_moved_this_step],
            "enemy_paused_this_step": [bool(value) for value in self.enemy_paused_this_step],
        }

    def _get_observation(self) -> np.ndarray:
        observation = super()._get_observation().copy()
        # Deterministic patrol phase is not defined for independent random
        # walkers. Preserve the established 24-value DQN interface without
        # leaking a fabricated clock/phase signal.
        observation[-2:] = 0.0
        if observation.shape != (OBSERVATION_DIM,):
            raise RuntimeError(f"Unexpected observation shape: {observation.shape}")
        return observation.astype(np.float32, copy=False)

    def _advance_enemies(self) -> None:
        """Advance each independent random walker by at most one cell."""
        self.enemy_moved_this_step = [False] * self.enemy_count
        self.enemy_paused_this_step = [False] * self.enemy_count

        # Randomize update order to avoid a permanent priority for low indices.
        update_order = list(range(self.enemy_count))
        self.rng.shuffle(update_order)
        occupied = set(self.moving_obstacle_positions)

        for enemy_index in update_order:
            current = self.moving_obstacle_positions[enemy_index]
            interval = int(self.enemy_speed_intervals[enemy_index])

            if self.enemy_pause_remaining[enemy_index] > 0:
                self.enemy_pause_remaining[enemy_index] -= 1
                self.enemy_paused_this_step[enemy_index] = True
                # Do not allow a long pause to accumulate overdue move events.
                if self.steps >= self.enemy_next_move_steps[enemy_index]:
                    self.enemy_next_move_steps[enemy_index] = self.steps + interval
                continue

            if self.steps < self.enemy_next_move_steps[enemy_index]:
                continue

            # This enemy has reached its next movement opportunity.
            self.enemy_next_move_steps[enemy_index] = self.steps + interval
            if self.rng.random() < self.pause_probability:
                pause_steps = int(self.rng.integers(
                    self.pause_duration[0], self.pause_duration[1] + 1
                ))
                self.enemy_pause_remaining[enemy_index] = pause_steps - 1
                self.enemy_paused_this_step[enemy_index] = True
                continue

            occupied.discard(current)
            row, col = current
            candidates = []
            for dr, dc in ACTIONS.values():
                candidate = (row + dr, col + dc)
                if (
                    self._in_bounds(candidate)
                    and candidate not in self.static_walls
                    and candidate not in occupied
                    and candidate != self.agent_pos
                    and candidate != self.goal_pos
                ):
                    candidates.append(candidate)

            if candidates:
                selected = candidates[int(self.rng.integers(len(candidates)))]
                self.moving_obstacle_positions[enemy_index] = selected
                self.enemy_moved_this_step[enemy_index] = selected != current
            occupied.add(self.moving_obstacle_positions[enemy_index])

    def step(self, action: int):
        """Advance the agent, independent enemies, and scheduled moving goal."""
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

        terminated = self.agent_pos == self.goal_pos
        goal_moved = False
        self.enemy_moved_this_step = [False] * self.enemy_count
        self.enemy_paused_this_step = [False] * self.enemy_count

        if not terminated:
            # Preserve version 03's transition order: agent first, dynamic
            # obstacles second, and the goal last on its configured cadence.
            self._advance_enemies()
            if self.steps % self.goal_move_interval == 0:
                goal_moved = self._advance_goal()

        truncated = self.steps >= self.max_steps and not terminated
        new_potential = 0.0 if terminated else self._potential(self.agent_pos)
        base_reward = self.goal_reward if terminated else self.step_penalty
        if blocked_reason is not None:
            base_reward += self.blocked_penalty
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
            "moving_obstacle_positions": self.enemy_positions,
            "steps": self.steps,
            "base_reward": float(base_reward),
            "shaping_reward": float(shaping_reward),
            "enemy_count": self.enemy_count,
            "enemies_moved_count": int(sum(self.enemy_moved_this_step)),
            "enemies_paused_count": int(sum(self.enemy_paused_this_step)),
        }
        info.update(self._enemy_info())
        self._last_info = info
        return self._get_observation(), reward, terminated, truncated, info

    def render(self, ax=None, show_sensor: bool = False):
        """Render the full map and distinguish enemy speeds and pause states."""
        owns_axes = ax is None
        if owns_axes:
            _, ax = plt.subplots(figsize=(7.5, 7.5))

        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        for row, col in self.static_walls:
            grid[row, col] = 1
        ax.imshow(
            grid,
            cmap=ListedColormap(["#f4f5f7", "#3d4755"]),
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )

        ar, ac = self.agent_pos
        gr, gc = self.goal_pos
        ax.scatter(ac, ar, s=165, color="#2878b5", edgecolor="white", linewidth=1.0,
                   label="Agent", zorder=6)
        ax.scatter(gc, gr, s=245, color="#2b9a62", marker="*", edgecolor="white",
                   linewidth=0.7, label="Goal", zorder=6)

        palette = plt.get_cmap("plasma")
        max_interval = max(self.default_enemy_speed_intervals)
        for index, (row, col) in enumerate(self.moving_obstacle_positions):
            interval = int(self.enemy_speed_intervals[index])
            color = palette(0.15 + 0.75 * (interval - 1) / max(max_interval - 1, 1))
            paused = self.enemy_pause_remaining[index] > 0 or self.enemy_paused_this_step[index]
            marker = "s" if paused else "o"
            ax.scatter(
                col, row, s=145, color=color, marker=marker,
                edgecolor="#222222", linewidth=0.8,
                label=f"Enemy {index + 1} (every {interval} step{'s' if interval > 1 else ''})",
                zorder=5,
            )
            ax.text(col, row, f"E{index + 1}", ha="center", va="center",
                    fontsize=7, color="white", weight="bold", zorder=7)

        if show_sensor and self.sensor_radius == 1:
            from matplotlib.patches import Polygon
            vertices = [
                (ac - 0.5, ar - 1.5), (ac + 0.5, ar - 1.5),
                (ac + 0.5, ar - 0.5), (ac + 1.5, ar - 0.5),
                (ac + 1.5, ar + 0.5), (ac + 0.5, ar + 0.5),
                (ac + 0.5, ar + 1.5), (ac - 0.5, ar + 1.5),
                (ac - 0.5, ar + 0.5), (ac - 1.5, ar + 0.5),
                (ac - 1.5, ar - 0.5), (ac - 0.5, ar - 0.5),
            ]
            ax.add_patch(Polygon(vertices, closed=True, fill=False,
                                 edgecolor="#2878b5", linewidth=1.8,
                                 linestyle="--", zorder=8))

        ax.set_xticks(range(self.grid_size))
        ax.set_yticks(range(self.grid_size))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(True, linewidth=0.45, alpha=0.45)
        ax.set_xlim(-0.5, self.grid_size - 0.5)
        ax.set_ylim(self.grid_size - 0.5, -0.5)
        ax.set_aspect("equal")
        ax.set_title(f"Stochastic Enemies | Step {self.steps} | Squares = paused")
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7, framealpha=0.92)
        if owns_axes:
            plt.tight_layout()
            plt.show()
        return ax
