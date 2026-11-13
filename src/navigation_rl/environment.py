"""Partially observable dynamic grid-world for autonomous navigation.

The target is visible. Fixed walls and five moving patrol obstacles are sensed
only in the four cardinally adjacent cells (Manhattan distance one). Patrols
follow deterministic closed paths;
the observation includes their global phase so that the environment dynamics
are time-aware and reproducible.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

GRID_SIZE = 20
SENSOR_RADIUS = 1
MAX_STEPS = 180
START_POSITION = (0, 0)
MOVE_INTERVAL = 1

# A fixed, connected maze layout. Keeping the layout fixed in the baseline
# isolates the effect of goal variation and local sensing.
STATIC_WALLS = frozenset({
    (2, 4), (2, 5), (3, 5), (4, 5), (4, 6),
    (6, 2), (7, 2), (8, 2), (8, 3), (8, 4),
    (5, 9), (6, 9), (6, 10), (6, 11),
    (2, 14), (3, 14), (4, 14), (4, 13),
    (10, 7), (11, 7), (12, 7), (12, 8), (12, 9),
    (8, 16), (9, 16), (10, 16), (10, 15),
    (15, 3), (16, 3), (16, 4), (16, 5),
    (14, 11), (15, 11), (15, 12), (16, 12),
    (17, 8), (17, 9),
})

# Five patrol obstacles move one cell per environment step along closed loops.
# These five routes are disjoint and do not overlap fixed walls.
MOVING_OBSTACLE_PATHS = (
    ((1, 8), (1, 9), (2, 9), (3, 9), (3, 8), (2, 8)),
    ((9, 4), (9, 5), (10, 5), (11, 5), (11, 4), (10, 4)),
    ((6, 13), (6, 14), (7, 14), (8, 14), (8, 13), (7, 13)),
    ((12, 13), (12, 14), (13, 14), (14, 14), (14, 13), (13, 13)),
    ((13, 1), (13, 2), (14, 2), (15, 2), (15, 1), (14, 1)),
)

ACTIONS = {
    0: (-1, 0),  # up
    1: (1, 0),   # down
    2: (0, -1),  # left
    3: (0, 1),   # right
}
ACTION_NAMES = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}

OBSERVATION_DIM = 4 + 2 * (2 * SENSOR_RADIUS + 1) ** 2 + 2
NUM_ACTIONS = len(ACTIONS)
PHASE_PERIOD = len(MOVING_OBSTACLE_PATHS[0])

ENV_CONFIG = {
    "grid_size": GRID_SIZE,
    "sensor_radius": SENSOR_RADIUS,
    "max_steps": MAX_STEPS,
    "start_position": START_POSITION,
    "static_walls": STATIC_WALLS,
    "moving_obstacle_paths": MOVING_OBSTACLE_PATHS,
    "move_interval": MOVE_INTERVAL,
}


class NavigationGridWorld:
    """Grid navigation environment with local obstacle sensing.

    API follows the Gymnasium convention:
      reset(seed=None) -> (observation, info)
      step(action) -> (observation, reward, terminated, truncated, info)

    Observation layout (24 float32 values by default):
      1. normalized agent row/column and visible goal row/column (4 values)
      2. static-wall occupancy in a 3x3 encoding (9 values)
      3. moving-obstacle occupancy in a 3x3 encoding (9 values)
      4. sine/cosine of patrol phase (2 values)

    Only the center and four cardinal neighbors are observable. The four
    diagonal entries in each 3x3 encoding are always zero (not observed).
    Out-of-grid cardinal neighbors are marked as static walls.
    """

    ACTIONS = ACTIONS
    ACTION_NAMES = ACTION_NAMES
    observation_dim = OBSERVATION_DIM
    num_actions = NUM_ACTIONS

    def __init__(
        self,
        grid_size: int = GRID_SIZE,
        sensor_radius: int = SENSOR_RADIUS,
        max_steps: int = MAX_STEPS,
        start_position: tuple[int, int] = START_POSITION,
        static_walls: Iterable[tuple[int, int]] = STATIC_WALLS,
        moving_obstacle_paths: Iterable[Iterable[tuple[int, int]]] = MOVING_OBSTACLE_PATHS,
        move_interval: int = MOVE_INTERVAL,
        gamma: float = 0.99,
        shaping_scale: float = 0.5,
        step_penalty: float = -0.05,
        blocked_penalty: float = -0.75,
        goal_reward: float = 20.0,
    ) -> None:
        self.grid_size = int(grid_size)
        self.sensor_radius = int(sensor_radius)
        self.max_steps = int(max_steps)
        self.start_position = tuple(start_position)
        self.static_walls = frozenset(tuple(p) for p in static_walls)
        self.moving_obstacle_paths = tuple(
            tuple(tuple(p) for p in path) for path in moving_obstacle_paths
        )
        self.move_interval = int(move_interval)
        self.gamma = float(gamma)
        self.shaping_scale = float(shaping_scale)
        self.step_penalty = float(step_penalty)
        self.blocked_penalty = float(blocked_penalty)
        self.goal_reward = float(goal_reward)

        if self.grid_size < 3:
            raise ValueError("grid_size must be at least 3")
        if self.sensor_radius < 1:
            raise ValueError("sensor_radius must be at least 1")
        if self.max_steps <= 0 or self.move_interval <= 0:
            raise ValueError("max_steps and move_interval must be positive")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if not 0.0 <= self.shaping_scale:
            raise ValueError("shaping_scale must be non-negative")
        if not self._in_bounds(self.start_position):
            raise ValueError("start_position must be inside the grid")
        if self.start_position in self.static_walls:
            raise ValueError("start_position cannot overlap a fixed wall")

        for wall in self.static_walls:
            if not self._in_bounds(wall):
                raise ValueError(f"Fixed wall outside grid: {wall}")

        occupied_path_cells: set[tuple[int, int]] = set()
        for path in self.moving_obstacle_paths:
            if len(path) < 2:
                raise ValueError("Each moving-obstacle path needs at least two cells")
            if any(not self._in_bounds(p) for p in path):
                raise ValueError("A moving-obstacle path contains an out-of-bounds cell")
            if any(p in self.static_walls for p in path):
                raise ValueError("A moving-obstacle path overlaps a fixed wall")
            if occupied_path_cells.intersection(path):
                raise ValueError("Moving-obstacle paths must not overlap")
            for current, following in zip(path, path[1:] + path[:1]):
                if self._manhattan(current, following) != 1:
                    raise ValueError("Moving-obstacle routes must contain adjacent steps")
            occupied_path_cells.update(path)

        self._path_cells = frozenset(occupied_path_cells)
        self._reachable_from_start = self._reachable_cells(self.start_position)
        self._goal_candidates = sorted(
            self._reachable_from_start - self.static_walls - self._path_cells
            - {self.start_position}
        )
        if not self._goal_candidates:
            raise ValueError("No reachable goal positions are available")

        # Episode state is initialized by reset().
        self.rng = np.random.default_rng()
        self.steps = 0
        self.agent_pos = self.start_position
        self.goal_pos = self._goal_candidates[-1]
        self.patrol_indices = [0] * len(self.moving_obstacle_paths)
        self.moving_obstacle_positions = [path[0] for path in self.moving_obstacle_paths]
        self._distance_to_goal: dict[tuple[int, int], int] = {}
        self._last_info: dict = {}

    @staticmethod
    def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _in_bounds(self, pos: tuple[int, int]) -> bool:
        return 0 <= pos[0] < self.grid_size and 0 <= pos[1] < self.grid_size

    def _reachable_cells(self, start: tuple[int, int]) -> set[tuple[int, int]]:
        queue = deque([start])
        visited = {start}
        while queue:
            row, col = queue.popleft()
            for dr, dc in ACTIONS.values():
                nxt = (row + dr, col + dc)
                if (
                    self._in_bounds(nxt)
                    and nxt not in self.static_walls
                    and nxt not in visited
                ):
                    visited.add(nxt)
                    queue.append(nxt)
        return visited

    def _build_goal_distances(self, goal: tuple[int, int]) -> dict[tuple[int, int], int]:
        """Shortest-path distances on the fixed map, ignoring moving patrols."""
        distances = {goal: 0}
        queue = deque([goal])
        while queue:
            row, col = queue.popleft()
            for dr, dc in ACTIONS.values():
                nxt = (row + dr, col + dc)
                if (
                    self._in_bounds(nxt)
                    and nxt not in self.static_walls
                    and nxt not in distances
                ):
                    distances[nxt] = distances[(row, col)] + 1
                    queue.append(nxt)
        return distances

    def _potential(self, position: tuple[int, int]) -> float:
        distance = self._distance_to_goal.get(position)
        if distance is None:
            distance = self._manhattan(position, self.goal_pos)
        return -self.shaping_scale * float(distance)

    def _patrol_phase(self) -> int:
        return (self.steps // self.move_interval) % PHASE_PERIOD

    def _get_observation(self) -> np.ndarray:
        patch_size = 2 * self.sensor_radius + 1
        static_patch = np.zeros((patch_size, patch_size), dtype=np.float32)
        moving_patch = np.zeros((patch_size, patch_size), dtype=np.float32)
        row0, col0 = self.agent_pos
        moving_set = set(self.moving_obstacle_positions)

        for patch_row, dr in enumerate(range(-self.sensor_radius, self.sensor_radius + 1)):
            for patch_col, dc in enumerate(range(-self.sensor_radius, self.sensor_radius + 1)):
                # One-step sensing means Manhattan distance <= 1.
                # Diagonal cells are outside the sensor's field of view.
                if abs(dr) + abs(dc) > self.sensor_radius:
                    continue
                cell = (row0 + dr, col0 + dc)
                if not self._in_bounds(cell) or cell in self.static_walls:
                    static_patch[patch_row, patch_col] = 1.0
                if cell in moving_set:
                    moving_patch[patch_row, patch_col] = 1.0

        scale = float(max(self.grid_size - 1, 1))
        phase_angle = 2.0 * math.pi * self._patrol_phase() / PHASE_PERIOD
        features = np.concatenate((
            np.asarray([
                self.agent_pos[0] / scale,
                self.agent_pos[1] / scale,
                self.goal_pos[0] / scale,
                self.goal_pos[1] / scale,
            ], dtype=np.float32),
            static_patch.reshape(-1),
            moving_patch.reshape(-1),
            np.asarray([math.sin(phase_angle), math.cos(phase_angle)], dtype=np.float32),
        ))
        if features.shape != (OBSERVATION_DIM,):
            raise RuntimeError(f"Unexpected observation shape: {features.shape}")
        return features.astype(np.float32, copy=False)

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))
        self.steps = 0
        self.agent_pos = self.start_position
        self.patrol_indices = [0] * len(self.moving_obstacle_paths)
        self.moving_obstacle_positions = [path[0] for path in self.moving_obstacle_paths]

        requested_goal = (options or {}).get("goal_position")
        if requested_goal is not None:
            goal = tuple(requested_goal)
            if goal not in self._goal_candidates:
                raise ValueError("Requested goal must be a reachable free cell outside patrol routes")
            self.goal_pos = goal
        else:
            self.goal_pos = self._goal_candidates[int(self.rng.integers(len(self._goal_candidates)))]

        self._distance_to_goal = self._build_goal_distances(self.goal_pos)
        observation = self._get_observation()
        info = {
            "agent_position": self.agent_pos,
            "goal_position": self.goal_pos,
            "moving_obstacle_positions": list(self.moving_obstacle_positions),
            "reason": None,
        }
        self._last_info = info
        return observation, info

    def _advance_patrols(self) -> None:
        occupied_by_other = set(self.moving_obstacle_positions)
        new_positions = []
        new_indices = self.patrol_indices.copy()
        for index, path in enumerate(self.moving_obstacle_paths):
            current = self.moving_obstacle_positions[index]
            occupied_by_other.discard(current)
            next_index = (self.patrol_indices[index] + 1) % len(path)
            candidate = path[next_index]
            if candidate != self.agent_pos and candidate not in occupied_by_other and candidate not in new_positions:
                new_indices[index] = next_index
                new_positions.append(candidate)
            else:
                new_positions.append(current)
                new_indices[index] = self.patrol_indices[index]
            occupied_by_other.add(new_positions[-1])
        self.patrol_indices = new_indices
        self.moving_obstacle_positions = new_positions

    def step(self, action: int):
        if not isinstance(action, (int, np.integer)) or int(action) not in ACTIONS:
            raise ValueError(f"action must be one of {sorted(ACTIONS)}")

        action = int(action)
        self.steps += 1
        old_position = self.agent_pos
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
        truncated = self.steps >= self.max_steps and not terminated

        # Patrols move after the agent. A patrol waits rather than entering the
        # agent's cell, so a moving obstacle never teleports through the agent.
        if not terminated:
            if self.steps % self.move_interval == 0:
                self._advance_patrols()

        new_potential = 0.0 if terminated else self._potential(self.agent_pos)
        base_reward = self.goal_reward if terminated else self.step_penalty
        if blocked_reason is not None:
            base_reward += self.blocked_penalty
        # Potential-based shaping: F(s,s') = gamma*Phi(s') - Phi(s).
        # Keep gamma synchronized with DQNConfig.gamma for policy invariance.
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
            "moving_obstacle_positions": list(self.moving_obstacle_positions),
            "steps": self.steps,
            "base_reward": float(base_reward),
            "shaping_reward": float(shaping_reward),
        }
        self._last_info = info
        return self._get_observation(), reward, terminated, truncated, info

    def render(self, ax=None, show_sensor: bool = False):
        """Render the full map for inspection (not the agent's observation)."""
        owns_axes = ax is None
        if owns_axes:
            _, ax = plt.subplots(figsize=(7, 7))

        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        for row, col in self.static_walls:
            grid[row, col] = 1
        for row, col in self.moving_obstacle_positions:
            grid[row, col] = 2
        cmap = ListedColormap(["#f4f5f7", "#3d4755", "#e39b36"])
        ax.imshow(grid, cmap=cmap, vmin=0, vmax=2)

        ar, ac = self.agent_pos
        gr, gc = self.goal_pos
        ax.scatter(ac, ar, s=150, color="#2878b5", label="Agent", zorder=5)
        ax.scatter(gc, gr, s=230, color="#2b9a62", marker="*", label="Goal", zorder=5)
        if self.moving_obstacle_positions:
            mr = [p[0] for p in self.moving_obstacle_positions]
            mc = [p[1] for p in self.moving_obstacle_positions]
            ax.scatter(mc, mr, s=110, color="#cf4545", marker="X", label="Moving obstacles", zorder=4)

        if show_sensor:
            from matplotlib.patches import Polygon
            # Outline the agent cell plus its four cardinal neighbors.
            if self.sensor_radius == 1:
                vertices = [
                    (ac - 0.5, ar - 1.5), (ac + 0.5, ar - 1.5),
                    (ac + 0.5, ar - 0.5), (ac + 1.5, ar - 0.5),
                    (ac + 1.5, ar + 0.5), (ac + 0.5, ar + 0.5),
                    (ac + 0.5, ar + 1.5), (ac - 0.5, ar + 1.5),
                    (ac - 0.5, ar + 0.5), (ac - 1.5, ar + 0.5),
                    (ac - 1.5, ar - 0.5), (ac - 0.5, ar - 0.5),
                ]
                ax.add_patch(Polygon(
                    vertices, closed=True, fill=False, edgecolor="#2878b5",
                    linewidth=1.8, linestyle="--",
                ))
        ax.set_xticks(range(self.grid_size))
        ax.set_yticks(range(self.grid_size))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(True, linewidth=0.45, alpha=0.45)
        ax.set_xlim(-0.5, self.grid_size - 0.5)
        ax.set_ylim(self.grid_size - 0.5, -0.5)
        ax.set_title(f"Autonomous Navigation | Step {self.steps}")
        ax.legend(loc="upper right", fontsize=8)
        if owns_axes:
            plt.tight_layout()
            plt.show()
        return ax


def encode_observation(observation, expected_dim: int = OBSERVATION_DIM) -> np.ndarray:
    """Validate and return a float32 observation vector.

    ``expected_dim`` defaults to the historical 24-value observation but can
    be set by agents using an environment-specific observation extension.
    """
    vector = np.asarray(observation, dtype=np.float32)
    expected_shape = (int(expected_dim),)
    if vector.shape != expected_shape:
        raise ValueError(f"Expected observation shape {expected_shape}, got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError("Observation contains non-finite values")
    return vector
