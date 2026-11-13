"""Recurrent Double-DQN utilities for partially observable navigation.

The replay buffer stores complete episodes and samples contiguous sequences.
A burn-in prefix reconstructs the recurrent state before each training window.
The environment is unchanged; only the value function gains temporal memory.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn


class DRQNNetwork(nn.Module):
    def __init__(self, observation_dim: int, num_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, observations: torch.Tensor, hidden=None):
        encoded = self.encoder(observations)
        recurrent, hidden = self.lstm(encoded, hidden)
        return self.head(recurrent), hidden


class EpisodicSequenceReplay:
    """Uniform episode/chunk replay with fixed-length burn-in and unroll."""
    def __init__(self, capacity_episodes: int, observation_dim: int, seed: int = 42):
        self.episodes = deque(maxlen=int(capacity_episodes))
        self.observation_dim = int(observation_dim)
        self.rng = np.random.default_rng(seed)
        self.transitions = 0

    def __len__(self):
        return self.transitions

    def add_episode(self, states, actions, rewards, dones):
        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.int64)
        rewards = np.asarray(rewards, dtype=np.float32)
        dones = np.asarray(dones, dtype=np.float32)
        length = len(actions)
        if states.shape != (length + 1, self.observation_dim):
            raise ValueError("Episode state sequence has an invalid shape")
        if rewards.shape != (length,) or dones.shape != (length,):
            raise ValueError("Episode transition arrays have inconsistent lengths")
        self.episodes.append({
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
        })
        self.transitions += length
        # Recalculate exact size after deque evictions.
        self.transitions = sum(len(ep["actions"]) for ep in self.episodes)

    def sample(self, batch_size: int, burn_in: int, sequence_length: int):
        eligible = [ep for ep in self.episodes if len(ep["actions"]) > burn_in]
        if not eligible:
            raise ValueError("No completed episodes are long enough for sequence replay")
        batch_states, batch_actions, batch_rewards, batch_dones, batch_masks = [], [], [], [], []
        for _ in range(batch_size):
            episode = eligible[int(self.rng.integers(len(eligible)))]
            length = len(episode["actions"])
            start = int(self.rng.integers(burn_in, length))
            prefix = episode["states"][start - burn_in:start]
            chunk_states = episode["states"][start:min(start + sequence_length, length) + 1]
            valid = min(sequence_length, length - start)
            if valid < sequence_length:
                pad_count = sequence_length + 1 - len(chunk_states)
                chunk_states = np.concatenate(
                    [chunk_states, np.repeat(chunk_states[-1:], pad_count, axis=0)], axis=0
                )
            states = np.concatenate([prefix, chunk_states], axis=0)
            actions = np.zeros(sequence_length, dtype=np.int64)
            rewards = np.zeros(sequence_length, dtype=np.float32)
            dones = np.ones(sequence_length, dtype=np.float32)
            mask = np.zeros(sequence_length, dtype=np.float32)
            actions[:valid] = episode["actions"][start:start + valid]
            rewards[:valid] = episode["rewards"][start:start + valid]
            dones[:valid] = episode["dones"][start:start + valid]
            mask[:valid] = 1.0
            batch_states.append(states)
            batch_actions.append(actions)
            batch_rewards.append(rewards)
            batch_dones.append(dones)
            batch_masks.append(mask)
        return (
            np.asarray(batch_states, dtype=np.float32),
            np.asarray(batch_actions, dtype=np.int64),
            np.asarray(batch_rewards, dtype=np.float32),
            np.asarray(batch_dones, dtype=np.float32),
            np.asarray(batch_masks, dtype=np.float32),
        )


@dataclass
class DRQNConfig:
    observation_dim: int = 96
    num_actions: int = 4
    hidden_dim: int = 128
    learning_rate: float = 1e-4
    gamma: float = 0.99
    batch_size: int = 16
    sequence_length: int = 12
    burn_in: int = 8
    replay_capacity_episodes: int = 500
    learning_starts: int = 1500
    train_frequency: int = 8
    target_update_interval: int = 400
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 250
    max_grad_norm: float = 10.0
    device: str | None = None


class DRQNAgent:
    def __init__(self, config: DRQNConfig, seed: int = 42):
        self.config = config
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed + 17)
        self.device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.online = DRQNNetwork(config.observation_dim, config.num_actions, config.hidden_dim).to(self.device)
        self.target = DRQNNetwork(config.observation_dim, config.num_actions, config.hidden_dim).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config.learning_rate)
        self.loss_fn = nn.SmoothL1Loss(reduction="none")
        self.replay = EpisodicSequenceReplay(config.replay_capacity_episodes, config.observation_dim, seed + 31)
        self.environment_steps = 0
        self.gradient_steps = 0
        self.last_loss = np.nan

    def initial_hidden(self, batch_size: int = 1):
        return (
            torch.zeros(1, batch_size, self.config.hidden_dim, device=self.device),
            torch.zeros(1, batch_size, self.config.hidden_dim, device=self.device),
        )

    def epsilon_for_episode(self, episode: int) -> float:
        fraction = min(1.0, max(0, episode - 1) / max(1, self.config.epsilon_decay_episodes - 1))
        return self.config.epsilon_start + fraction * (self.config.epsilon_end - self.config.epsilon_start)

    @torch.no_grad()
    def act(self, observation, hidden, epsilon: float = 0.0, explore: bool = False):
        state = torch.as_tensor(np.asarray(observation, dtype=np.float32), device=self.device).view(1, 1, -1)
        q_values, next_hidden = self.online(state, hidden)
        if explore and self.rng.random() < epsilon:
            action = int(self.rng.integers(self.config.num_actions))
        else:
            action = int(q_values[0, -1].argmax().item())
        return action, next_hidden

    def learn(self):
        cfg = self.config
        arrays = self.replay.sample(cfg.batch_size, cfg.burn_in, cfg.sequence_length)
        states, actions, rewards, dones, masks = [torch.as_tensor(x, device=self.device) for x in arrays]
        burn_states = states[:, :cfg.burn_in]
        learn_states = states[:, cfg.burn_in:]

        # Reconstruct recurrent context without back-propagating through burn-in.
        with torch.no_grad():
            _, online_hidden = self.online(burn_states)
            _, target_hidden = self.target(burn_states)
        online_q, _ = self.online(learn_states, tuple(x.detach() for x in online_hidden))
        with torch.no_grad():
            target_q, _ = self.target(learn_states, tuple(x.detach() for x in target_hidden))
            next_actions = online_q[:, 1:].argmax(dim=-1, keepdim=True)
            next_values = target_q[:, 1:].gather(-1, next_actions).squeeze(-1)
            targets = rewards + cfg.gamma * (1.0 - dones) * next_values

        current_q = online_q[:, :-1].gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        element_loss = self.loss_fn(current_q, targets)
        loss = (element_loss * masks).sum() / masks.sum().clamp_min(1.0)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), cfg.max_grad_norm)
        self.optimizer.step()
        self.gradient_steps += 1
        if self.gradient_steps % cfg.target_update_interval == 0:
            self.target.load_state_dict(self.online.state_dict())
        self.last_loss = float(loss.item())
        return self.last_loss

    def save(self, path: str | Path, extra: dict[str, Any] | None = None, include_replay: bool = True):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "online_state_dict": self.online.state_dict(),
            "target_state_dict": self.target.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config.__dict__,
            "environment_steps": self.environment_steps,
            "gradient_steps": self.gradient_steps,
            "rng_state": self.rng.bit_generator.state,
            "replay_rng_state": self.replay.rng.bit_generator.state,
            "extra": extra or {},
        }
        if include_replay:
            payload["replay_episodes"] = list(self.replay.episodes)
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str | Path, map_location: str | None = None):
        checkpoint = torch.load(path, map_location=map_location or "cpu", weights_only=False)
        cfg = DRQNConfig(**checkpoint["config"])
        agent = cls(cfg, seed=42)
        agent.online.load_state_dict(checkpoint["online_state_dict"])
        agent.target.load_state_dict(checkpoint["target_state_dict"])
        agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        agent.environment_steps = int(checkpoint.get("environment_steps", 0))
        agent.gradient_steps = int(checkpoint.get("gradient_steps", 0))
        if "rng_state" in checkpoint:
            agent.rng.bit_generator.state = checkpoint["rng_state"]
        if "replay_rng_state" in checkpoint:
            agent.replay.rng.bit_generator.state = checkpoint["replay_rng_state"]
        for episode in checkpoint.get("replay_episodes", []):
            agent.replay.episodes.append(episode)
        agent.replay.transitions = sum(len(ep["actions"]) for ep in agent.replay.episodes)
        return agent, checkpoint.get("extra", {})


def train_drqn(
    env_factory, config: DRQNConfig, episodes: int = 1000, seed: int = 42,
    log_interval: int = 50, progress: bool = True, agent: DRQNAgent | None = None,
    start_episode: int = 0, history: pd.DataFrame | None = None,
    checkpoint_path: str | Path | None = None, checkpoint_interval: int = 100,
):
    agent = agent or DRQNAgent(config, seed=seed)
    history_rows = history.to_dict("records") if history is not None and not history.empty else []
    started = time.perf_counter()
    for episode in range(start_episode + 1, episodes + 1):
        env = env_factory()
        obs, _ = env.reset(seed=seed + episode)
        hidden = agent.initial_hidden()
        epsilon = agent.epsilon_for_episode(episode)
        states = [np.asarray(obs, dtype=np.float32).copy()]
        actions, rewards, dones = [], [], []
        total_return = 0.0
        terminated = truncated = False
        info = {}
        while not (terminated or truncated):
            action, hidden = agent.act(obs, hidden, epsilon=epsilon, explore=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            actions.append(action)
            rewards.append(float(reward))
            dones.append(float(bool(terminated or truncated)))
            states.append(np.asarray(next_obs, dtype=np.float32).copy())
            total_return += float(reward)
            obs = next_obs
            agent.environment_steps += 1
            if agent.environment_steps >= config.learning_starts and agent.environment_steps % config.train_frequency == 0:
                if len(agent.replay.episodes) > 0:
                    try:
                        agent.learn()
                    except ValueError:
                        pass
        agent.replay.add_episode(states, actions, rewards, dones)
        row = {
            "episode": episode,
            "episode_return": total_return,
            "episode_length": len(actions),
            "success": bool(info.get("success", False)),
            "enemy_collision": bool(info.get("enemy_collision", False)),
            "epsilon": epsilon,
            "mean_loss": agent.last_loss,
            "environment_steps": agent.environment_steps,
            "gradient_steps": agent.gradient_steps,
        }
        history_rows.append(row)
        if progress and episode % log_interval == 0:
            recent = history_rows[-log_interval:]
            print(
                f"DRQN episode {episode:4d}/{episodes} | "
                f"success(last {log_interval})={np.mean([x['success'] for x in recent]):.1%} | "
                f"return={np.mean([x['episode_return'] for x in recent]):.2f} | "
                f"epsilon={epsilon:.3f} | updates={agent.gradient_steps}"
            )
        if checkpoint_path is not None and checkpoint_interval > 0 and episode % checkpoint_interval == 0:
            checkpoint_path = Path(checkpoint_path)
            agent.save(checkpoint_path, extra={"completed_episodes": episode}, include_replay=True)
            pd.DataFrame(history_rows).to_csv(checkpoint_path.with_name("training_progress.csv"), index=False)
    history_frame = pd.DataFrame(history_rows)
    elapsed = time.perf_counter() - started
    return agent, history_frame, elapsed


@torch.no_grad()
def evaluate_drqn(agent: DRQNAgent, env_factory, seeds, max_steps: int = 500):
    rows = []
    for seed in seeds:
        env = env_factory()
        obs, _ = env.reset(seed=int(seed))
        hidden = agent.initial_hidden()
        total_return = 0.0
        info = {}
        for _ in range(max_steps):
            action, hidden = agent.act(obs, hidden, epsilon=0.0, explore=False)
            obs, reward, terminated, truncated, info = env.step(action)
            total_return += float(reward)
            if terminated or truncated:
                break
        rows.append({
            "method": "DRQN_range4_danger",
            "seed": int(seed),
            "success": int(bool(info.get("success", False))),
            "collision": int(bool(info.get("enemy_collision", False))),
            "steps": int(env.steps),
            "return": total_return,
            "reason": info.get("reason") or "time_limit",
        })
    return pd.DataFrame(rows)
