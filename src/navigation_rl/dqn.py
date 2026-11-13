"""Vanilla DQN implementation and experiment utilities for NavigationGridWorld."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
import random
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm.auto import tqdm

from .environment import NavigationGridWorld, OBSERVATION_DIM, NUM_ACTIONS, MAX_STEPS, encode_observation


@dataclass
class DQNConfig:
    observation_dim: int = OBSERVATION_DIM
    num_actions: int = NUM_ACTIONS
    learning_rate: float = 1e-4
    gamma: float = 0.99
    replay_capacity: int = 100_000
    batch_size: int = 128
    learning_starts: int = 3_000
    train_frequency: int = 4
    target_update_interval: int = 2_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 300_000
    hidden_dim: int = 256
    max_grad_norm: float = 10.0
    device: str | None = None

    def validate(self) -> None:
        if self.observation_dim <= 0 or self.num_actions <= 0:
            raise ValueError("observation_dim and num_actions must be positive")
        if self.learning_rate <= 0 or not 0 < self.gamma <= 1:
            raise ValueError("learning_rate must be positive and gamma in (0, 1]")
        if self.replay_capacity < self.batch_size:
            raise ValueError("replay_capacity must be >= batch_size")
        if self.batch_size <= 0 or self.learning_starts < self.batch_size:
            raise ValueError("learning_starts must be >= batch_size")
        if self.train_frequency <= 0 or self.target_update_interval <= 0:
            raise ValueError("update intervals must be positive")
        if not 0 <= self.epsilon_end <= self.epsilon_start <= 1:
            raise ValueError("epsilon values must satisfy 0 <= end <= start <= 1")
        if self.epsilon_decay_steps <= 0 or self.hidden_dim <= 0:
            raise ValueError("epsilon_decay_steps and hidden_dim must be positive")


@dataclass
class TrainingConfig:
    episodes: int = 20_000
    seed: int = 42
    evaluation_episodes: int = 200
    evaluation_seed: int = 50_000
    max_steps: int = 180
    log_interval: int = 100
    checkpoint_interval: int = 5_000
    output_dir: str = "results"
    show_progress: bool = True

    def validate(self) -> None:
        if self.episodes <= 0 or self.evaluation_episodes <= 0:
            raise ValueError("episodes and evaluation_episodes must be positive")
        if self.max_steps <= 0 or self.log_interval <= 0:
            raise ValueError("max_steps and log_interval must be positive")


class ReplayBuffer:
    """Preallocated replay buffer with uniform random sampling."""

    def __init__(self, capacity: int, observation_dim: int, seed: int | None = None):
        if capacity <= 0 or observation_dim <= 0:
            raise ValueError("capacity and observation_dim must be positive")
        self.capacity = int(capacity)
        self.observation_dim = int(observation_dim)
        self.states = np.zeros((capacity, observation_dim), dtype=np.float32)
        self.next_states = np.zeros_like(self.states)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.size

    def add(self, state, action: int, reward: float, next_state, terminated: bool) -> None:
        state = encode_observation(state, expected_dim=self.observation_dim)
        next_state = encode_observation(next_state, expected_dim=self.observation_dim)
        if state.shape != (self.observation_dim,) or next_state.shape != (self.observation_dim,):
            raise ValueError("Transition observation dimension does not match replay buffer")
        idx = self.position
        self.states[idx] = state
        self.actions[idx] = int(action)
        self.rewards[idx] = float(reward)
        self.next_states[idx] = next_state
        self.terminated[idx] = float(bool(terminated))
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device):
        if self.size < batch_size:
            raise ValueError("Not enough transitions to sample a batch")
        indices = self.rng.choice(self.size, size=batch_size, replace=False)
        return (
            torch.as_tensor(self.states[indices], device=device),
            torch.as_tensor(self.actions[indices], device=device),
            torch.as_tensor(self.rewards[indices, None], device=device),
            torch.as_tensor(self.next_states[indices], device=device),
            torch.as_tensor(self.terminated[indices, None], device=device),
        )


class QNetwork(nn.Module):
    def __init__(self, observation_dim: int, num_actions: int, hidden_dim: int = 256):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class DQNAgent:
    """Vanilla DQN: online/target networks, replay, epsilon-greedy, Huber loss."""

    def __init__(self, config: DQNConfig | None = None, seed: int = 42):
        self.config = config or DQNConfig()
        self.config.validate()
        self.seed = int(seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        self.device = torch.device(
            self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.action_rng = np.random.default_rng(self.seed + 17)
        self.replay_buffer = ReplayBuffer(
            self.config.replay_capacity,
            self.config.observation_dim,
            seed=self.seed + 31,
        )
        self.online_network = QNetwork(
            self.config.observation_dim, self.config.num_actions, self.config.hidden_dim
        ).to(self.device)
        self.target_network = QNetwork(
            self.config.observation_dim, self.config.num_actions, self.config.hidden_dim
        ).to(self.device)
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()
        self.optimizer = torch.optim.Adam(
            self.online_network.parameters(), lr=self.config.learning_rate
        )
        self.loss_function = nn.SmoothL1Loss()
        self.environment_steps = 0
        self.gradient_steps = 0
        self.last_loss: float | None = None

    @property
    def epsilon(self) -> float:
        fraction = min(1.0, self.environment_steps / self.config.epsilon_decay_steps)
        return self.config.epsilon_start + fraction * (
            self.config.epsilon_end - self.config.epsilon_start
        )

    def choose_action_with_info(
        self,
        observation,
        explore: bool = False,
        epsilon: float | None = None,
    ) -> tuple[int, dict[str, Any]]:
        """Choose an action using the default or an explicitly supplied epsilon.

        ``epsilon`` overrides the agent's step-based schedule for this decision
        only. It does not modify the agent's environment-step counter or config.
        """
        state = encode_observation(observation, expected_dim=self.config.observation_dim)
        scheduled_epsilon = float(self.epsilon if epsilon is None else epsilon)
        if not np.isfinite(scheduled_epsilon) or not 0.0 <= scheduled_epsilon <= 1.0:
            raise ValueError("epsilon must be a finite number in [0, 1]")
        if explore and self.action_rng.random() < scheduled_epsilon:
            action = int(self.action_rng.integers(self.config.num_actions))
            source = "RANDOM / epsilon exploration"
            effective_epsilon = scheduled_epsilon
        else:
            self.online_network.eval()
            with torch.no_grad():
                tensor = torch.as_tensor(state[None, :], device=self.device)
                action = int(self.online_network(tensor).argmax(dim=1).item())
            source = "GREEDY / DQN highest Q-value" if explore else "GREEDY / evaluation"
            effective_epsilon = scheduled_epsilon if explore else 0.0

        return action, {
            "epsilon": effective_epsilon,
            "scheduled_epsilon": scheduled_epsilon,
            "action_source": source,
            "exploration_enabled": bool(explore),
        }

    def choose_action(
        self,
        observation,
        explore: bool = False,
        epsilon: float | None = None,
    ) -> int:
        # Preserve the public API used by versions 02/03/04 while allowing an
        # optional per-decision override for experiments that need it.
        action, _ = self.choose_action_with_info(
            observation,
            explore=explore,
            epsilon=epsilon,
        )
        return action

    def store_transition(self, state, action, reward, next_state, terminated) -> float | None:
        self.replay_buffer.add(state, action, reward, next_state, terminated)
        self.environment_steps += 1
        loss = None
        if (
            len(self.replay_buffer) >= max(self.config.learning_starts, self.config.batch_size)
            and self.environment_steps % self.config.train_frequency == 0
        ):
            loss = self.learn()
        if self.environment_steps % self.config.target_update_interval == 0:
            self.target_network.load_state_dict(self.online_network.state_dict())
        return loss

    def learn(self) -> float:
        states, actions, rewards, next_states, terminated = self.replay_buffer.sample(
            self.config.batch_size, self.device
        )
        self.online_network.train()
        current_q = self.online_network(states).gather(1, actions.unsqueeze(1))
        with torch.no_grad():
            next_q = self.target_network(next_states).max(dim=1, keepdim=True).values
            targets = rewards + self.config.gamma * (1.0 - terminated) * next_q
        loss = self.loss_function(current_q, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_network.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        self.gradient_steps += 1
        self.last_loss = float(loss.detach().cpu().item())
        return self.last_loss

    def save(self, path: str | Path, extra: dict[str, Any] | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "online_network": self.online_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "dqn_config": asdict(self.config),
            "seed": self.seed,
            "environment_steps": self.environment_steps,
            "gradient_steps": self.gradient_steps,
            "extra": extra or {},
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str | Path, map_location: str | None = None) -> "DQNAgent":
        try:
            checkpoint = torch.load(path, map_location=map_location or "cpu", weights_only=False)
        except TypeError:  # Compatibility with older PyTorch releases.
            checkpoint = torch.load(path, map_location=map_location or "cpu")
        config = DQNConfig(**checkpoint["dqn_config"])
        agent = cls(config=config, seed=int(checkpoint.get("seed", 42)))
        agent.online_network.load_state_dict(checkpoint["online_network"])
        agent.target_network.load_state_dict(checkpoint["target_network"])
        agent.environment_steps = int(checkpoint.get("environment_steps", 0))
        agent.gradient_steps = int(checkpoint.get("gradient_steps", 0))
        return agent



def _capture_episode_frame(
    env: NavigationGridWorld,
    step: int,
    action: int | None,
    reward: float,
    episode_return: float,
    info: dict[str, Any],
    learning_return: float | None = None,
    action_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Capture the environment state and epsilon/action decision metadata."""
    action_names = getattr(env, "ACTION_NAMES", {})
    frame = {
        "step": int(step),
        "agent_position": [int(value) for value in env.agent_pos],
        "goal_position": [int(value) for value in env.goal_pos],
        "moving_obstacle_positions": [
            [int(value) for value in position]
            for position in env.moving_obstacle_positions
        ],
        "action": None if action is None else int(action),
        "action_name": "START" if action is None else str(action_names.get(int(action), int(action))),
        "reward": float(reward),
        "task_reward": float(info.get("task_reward", info.get("base_reward", reward))),
        "learning_reward": float(info.get("learning_reward", reward)),
        "episode_return": float(episode_return),
        "episode_learning_return": float(
            episode_return if learning_return is None else learning_return
        ),
        "blocked": bool(info.get("blocked", False)),
        "success": bool(info.get("success", False)),
        "terminated": bool(info.get("terminated", info.get("success", False))),
        "truncated": info.get("reason") == "time_limit",
        "base_reward": float(info.get("base_reward", reward)),
        "goal_shaping_reward": float(info.get("goal_shaping_reward", 0.0)),
        "danger_shaping_reward": float(info.get("danger_shaping_reward", 0.0)),
        "shaping_reward": float(info.get("shaping_reward", 0.0)),
        "enemy_danger_potential_before": float(info.get("enemy_danger_potential_before", 0.0)),
        "enemy_danger_potential_after": float(info.get("enemy_danger_potential_after", 0.0)),
        "blocked_reason": info.get("blocked_reason"),
        "enemy_collision": bool(info.get("enemy_collision", False)),
    }
    if action_metadata is None:
        if action is not None:
            raise RuntimeError(
                "A recorded Agent action is missing its epsilon/action-source metadata. "
                "Pass choose_action_with_info() metadata for every recorded step."
            )
    else:
        required_metadata = {"epsilon", "scheduled_epsilon", "action_source", "exploration_enabled"}
        missing_metadata = required_metadata.difference(action_metadata)
        if missing_metadata:
            raise RuntimeError(
                f"Incomplete action metadata while recording an episode: {sorted(missing_metadata)}"
            )
        frame.update({
            "epsilon": float(action_metadata["epsilon"]),
            "scheduled_epsilon": float(action_metadata["scheduled_epsilon"]),
            "action_source": str(action_metadata["action_source"]),
            "exploration_enabled": bool(action_metadata["exploration_enabled"]),
        })
    # Optional dynamics metadata is copied only when an environment exposes
    # it. Existing 02/03 episode records therefore retain their old schema.
    optional_sequence_fields = (
        "enemy_speed_intervals",
        "enemy_pause_remaining",
        "enemy_moved_this_step",
        "enemy_paused_this_step",
        # v05 records the internal enemy modes for faithful episode animation.
        "enemy_states",
        "enemy_last_seen_positions",
        "enemy_search_opportunities",
        "enemy_detecting_agent",
        "enemy_next_move_steps",
        "agent_enemy_memory_positions",
        "agent_enemy_memory_steps",
    )
    for key in optional_sequence_fields:
        if key in info:
            frame[key] = list(info[key])
    return frame


def _save_episode_record(
    env: NavigationGridWorld,
    output_dir: Path,
    episode: int,
    episode_seed: int,
    frames: list[dict[str, Any]],
    success: bool,
    final_info: dict[str, Any],
    episode_return: float,
    episode_learning_return: float | None = None,
) -> Path:
    """Persist selected training-episode frames without storing rendered images."""
    import json

    episode_dir = output_dir / "episodes"
    episode_dir.mkdir(parents=True, exist_ok=True)
    environment_metadata = {
        "class_name": type(env).__name__,
        "grid_size": int(env.grid_size),
        "static_walls": [
            [int(value) for value in position]
            for position in sorted(env.static_walls)
        ],
        "start_position": [int(value) for value in env.start_position],
        "max_steps": int(env.max_steps),
        "goal_move_interval": getattr(env, "goal_move_interval", None),
        "enemy_speed_intervals": list(getattr(env, "enemy_speed_intervals", [])),
        "pause_probability": getattr(env, "pause_probability", None),
        "pause_duration": list(getattr(env, "pause_duration", ())),
    }
    # Environment-specific metadata is included only when the environment
    # exposes it, keeping the existing v02-v04 record format compatible.
    for attribute in (
        "detection_range",
        "search_opportunities",
        "goal_reward",
        "step_penalty",
        "wall_penalty",
        "boundary_penalty",
        "enemy_blocked_penalty",
        "enemy_collision_penalty",
        "agent_enemy_detection_range",
        "goal_shaping_scale",
        "enemy_danger_radius",
        "enemy_danger_scale",
        "remember_seen_enemies",
        "enemy_memory_age_horizon",
    ):
        if hasattr(env, attribute):
            environment_metadata[attribute] = getattr(env, attribute)
    payload = {
        "schema_version": 1,
        "action_trace_version": 1,
        "episode": int(episode),
        "episode_seed": int(episode_seed),
        "policy": "epsilon_greedy_training_policy",
        "environment": environment_metadata,
        "success": bool(success),
        "episode_return": float(episode_return),
        "episode_learning_return": float(
            episode_return if episode_learning_return is None else episode_learning_return
        ),
        "episode_length": int(final_info.get("steps", max(len(frames) - 1, 0))),
        "final_reason": final_info.get("reason"),
        "frames": frames,
    }
    path = episode_dir / f"episode_{episode:06d}.json"
    temporary_path = path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary_path.replace(path)
    return path


def train_agent(
    agent: DQNAgent,
    env: NavigationGridWorld,
    config: TrainingConfig | None = None,
    record_episodes: Iterable[int] | None = None,
    epsilon_schedule: Callable[[int], float] | None = None,
) -> pd.DataFrame:
    """Train one DQN seed and return episode-level metrics.

    ``epsilon_schedule`` is an optional callable receiving the one-based
    episode number and returning the epsilon to use throughout that episode.
    When omitted, the agent's original step-based linear epsilon schedule is
    used unchanged.

    ``record_episodes`` optionally selects one-based training episode numbers
    whose actual step-by-step environment states are saved as JSON under
    ``<config.output_dir>/episodes``. Recording is disabled by default and
    does not change action selection, rewards, replay insertion, or updates.
    """
    config = config or TrainingConfig()
    config.validate()
    if epsilon_schedule is not None and not callable(epsilon_schedule):
        raise TypeError("epsilon_schedule must be callable or None")
    selected_episodes = set(int(value) for value in (record_episodes if record_episodes is not None else ()))
    if any(value <= 0 or value > config.episodes for value in selected_episodes):
        raise ValueError("record_episodes must contain episode numbers in [1, config.episodes]")
    if abs(env.gamma - agent.config.gamma) > 1e-12:
        raise ValueError("Environment shaping gamma must match DQNConfig.gamma")
    env.max_steps = config.max_steps

    rows: list[dict[str, float | int | bool]] = []
    recent_success = deque(maxlen=100)
    recent_return = deque(maxlen=100)
    iterator = tqdm(
        range(1, config.episodes + 1),
        desc="DQN training",
        unit="episode",
        disable=not config.show_progress,
    )
    for episode in iterator:
        episode_seed = config.seed + episode
        scheduled_epsilon = (
            float(agent.epsilon)
            if epsilon_schedule is None
            else float(epsilon_schedule(episode))
        )
        if not np.isfinite(scheduled_epsilon) or not 0.0 <= scheduled_epsilon <= 1.0:
            raise ValueError(
                f"epsilon_schedule({episode}) must return a finite value in [0, 1]"
            )
        observation, reset_info = env.reset(seed=episode_seed)
        should_record = episode in selected_episodes
        episode_frames: list[dict[str, Any]] = []
        total_reward = 0.0
        total_task_reward = 0.0
        blocked_actions = 0

        if should_record:
            episode_frames.append(_capture_episode_frame(
                env=env,
                step=0,
                action=None,
                reward=0.0,
                episode_return=0.0,
                info=reset_info,
                learning_return=0.0,
                action_metadata={
                    "epsilon": scheduled_epsilon,
                    "scheduled_epsilon": scheduled_epsilon,
                    "action_source": "NOT STARTED",
                    "exploration_enabled": True,
                },
            ))
        losses: list[float] = []
        terminated = truncated = False
        info: dict = {}

        while not (terminated or truncated):
            action, action_metadata = agent.choose_action_with_info(
                observation,
                explore=True,
                epsilon=scheduled_epsilon if epsilon_schedule is not None else None,
            )
            next_observation, reward, terminated, truncated, info = env.step(action)
            loss = agent.store_transition(
                observation, action, reward, next_observation, terminated
            )
            if loss is not None:
                losses.append(loss)
            total_reward += float(reward)
            total_task_reward += float(info.get("task_reward", reward))
            blocked_actions += int(info["blocked"])
            observation = next_observation
            if should_record:
                episode_frames.append(_capture_episode_frame(
                    env=env,
                    step=int(info.get("steps", len(episode_frames))),
                    action=action,
                    reward=float(reward),
                    episode_return=total_task_reward,
                    info=info,
                    learning_return=total_reward,
                    action_metadata=action_metadata,
                ))

        success = bool(info.get("success", False))
        if should_record:
            _save_episode_record(
                env=env,
                output_dir=Path(config.output_dir),
                episode=episode,
                episode_seed=episode_seed,
                frames=episode_frames,
                success=success,
                final_info=info,
                episode_return=total_task_reward,
                episode_learning_return=total_reward,
            )
        length = int(info.get("steps", 0))
        # Keep the historical step-based value when no custom schedule is used;
        # otherwise report the epsilon that actually governed this episode.
        logged_epsilon = scheduled_epsilon if epsilon_schedule is not None else agent.epsilon
        row = {
            "episode": episode,
            # Keep the public/final return unshaped in v05; the DQN still learns
            # from total_reward, which includes the configurable auxiliary terms.
            "episode_return": total_task_reward,
            "episode_learning_return": total_reward,
            "episode_length": length,
            "success": success,
            "blocked_actions": blocked_actions,
            "blocked_action_rate": blocked_actions / max(length, 1),
            "epsilon": logged_epsilon,
            "mean_loss": float(np.mean(losses)) if losses else np.nan,
            "environment_steps": agent.environment_steps,
            "gradient_steps": agent.gradient_steps,
        }
        rows.append(row)
        recent_success.append(float(success))
        recent_return.append(total_task_reward)

        if episode % config.log_interval == 0:
            iterator.set_postfix(
                success_100=f"{np.mean(recent_success):.1%}",
                return_100=f"{np.mean(recent_return):.2f}",
                epsilon=f"{logged_epsilon:.3f}",
            )

        if config.checkpoint_interval and episode % config.checkpoint_interval == 0:
            output_dir = Path(config.output_dir)
            agent.save(output_dir / f"dqn_episode_{episode}.pt", extra={"episode": episode})

    iterator.close()
    history = pd.DataFrame(rows)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history.to_csv(output_dir / "training_history.csv", index=False)
    agent.save(output_dir / "dqn_final.pt", extra={"episodes": config.episodes})
    return history


def evaluate_policy(
    agent: DQNAgent,
    num_episodes: int = 200,
    seed: int = 50_000,
    max_steps: int = MAX_STEPS,
) -> dict[str, float]:
    """Evaluate greedily on seeded, randomly sampled goals; no learning occurs."""
    if num_episodes <= 0:
        raise ValueError("num_episodes must be positive")
    env = NavigationGridWorld(gamma=agent.config.gamma, max_steps=max_steps)
    returns, lengths, successes, blocked_rates = [], [], [], []
    for episode in range(num_episodes):
        observation, _ = env.reset(seed=seed + episode)
        total_reward = 0.0
        blocked = 0
        terminated = truncated = False
        info: dict = {}
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
    return {
        "success_rate": float(np.mean(successes)),
        "average_return": float(np.mean(returns)),
        "average_steps": float(np.mean(lengths)),
        "blocked_action_rate": float(np.mean(blocked_rates)),
    }


def plot_training_history(history: pd.DataFrame, window: int = 100):
    """Plot task return, optional shaped learning return, success, and length."""
    if window <= 0:
        raise ValueError("window must be positive")
    if history.empty:
        raise ValueError("history is empty")

    specs = [("episode_return", "Task return", "Task return")]
    if "episode_learning_return" in history and not np.allclose(
        history["episode_return"].astype(float),
        history["episode_learning_return"].astype(float),
        rtol=1e-6,
        atol=1e-8,
        equal_nan=True,
    ):
        specs.append(("episode_learning_return", "Learning return (with shaping)", "Learning return"))
    specs.extend([
        ("success", "Success rate", "Success rate"),
        ("episode_length", "Episode length", "Steps"),
    ])

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(specs), figsize=(5.2 * len(specs), 4))
    axes = np.atleast_1d(axes)
    for ax, (column, title, ylabel) in zip(axes, specs):
        values = history[column].astype(float)
        smooth = values.rolling(window, min_periods=1).mean()
        ax.plot(history["episode"], smooth)
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        if column == "success":
            ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    return fig
