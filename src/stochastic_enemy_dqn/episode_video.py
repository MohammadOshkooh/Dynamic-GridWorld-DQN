"""Render recorded stochastic-enemy training episodes as GIF / optional MP4."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap


def render_stochastic_enemy_episode(
    record_path: str | Path,
    output_dir: str | Path | None = None,
    fps: int = 10,
    save_mp4: bool = True,
) -> dict[str, str | int | bool | None]:
    """Render one real recorded training episode, including enemy pauses."""
    record_path = Path(record_path)
    if fps <= 0:
        raise ValueError("fps must be a positive integer")
    if not record_path.is_file():
        raise FileNotFoundError(record_path)

    payload = json.loads(record_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported episode record schema in {record_path}")
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("Episode record has no frames")
    required = {"agent_position", "goal_position", "moving_obstacle_positions", "step"}
    for frame_index, frame in enumerate(frames):
        missing = required.difference(frame)
        if missing:
            raise ValueError(f"Frame {frame_index} is missing: {sorted(missing)}")

    metadata = payload["environment"]
    grid_size = int(metadata["grid_size"])
    walls = {tuple(position) for position in metadata["static_walls"]}
    episode = int(payload["episode"])
    intervals = metadata.get("enemy_speed_intervals") or frames[0].get("enemy_speed_intervals")
    if not intervals:
        raise ValueError("Record does not contain enemy speed metadata")
    intervals = [int(value) for value in intervals]

    output_dir = Path(output_dir) if output_dir is not None else record_path.parent / "videos"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"episode_{episode:06d}"
    gif_path = output_dir / f"{stem}.gif"
    mp4_path = output_dir / f"{stem}.mp4"

    background = np.zeros((grid_size, grid_size), dtype=np.uint8)
    for row, col in walls:
        background[row, col] = 1

    fig, ax = plt.subplots(figsize=(8.5, 8))
    ax.imshow(
        background,
        cmap=ListedColormap(["#f4f5f7", "#3d4755"]),
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    ax.set_xticks(range(grid_size))
    ax.set_yticks(range(grid_size))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, linewidth=0.45, alpha=0.45)
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(grid_size - 0.5, -0.5)
    ax.set_aspect("equal")

    agent_marker = ax.scatter([], [], s=165, color="#2878b5", marker="o",
                              edgecolor="white", linewidth=1.0, label="Agent", zorder=7)
    goal_marker = ax.scatter([], [], s=245, color="#2b9a62", marker="*",
                             edgecolor="white", linewidth=0.8, label="Moving goal", zorder=7)
    agent_trail, = ax.plot([], [], color="#2878b5", linewidth=1.8, alpha=0.75,
                           label="Agent path", zorder=4)
    goal_trail, = ax.plot([], [], color="#2b9a62", linewidth=1.2, linestyle="--",
                          alpha=0.55, label="Goal path", zorder=3)

    palette = plt.get_cmap("plasma")
    max_interval = max(intervals)
    colors = [
        palette(0.15 + 0.75 * (interval - 1) / max(max_interval - 1, 1))
        for interval in intervals
    ]
    enemy_markers = []
    enemy_labels = []
    for enemy_index, (interval, color) in enumerate(zip(intervals, colors), start=1):
        marker, = ax.plot([], [], linestyle="", marker="o", markersize=10,
                          markerfacecolor=color, markeredgecolor="#222222",
                          markeredgewidth=0.8, zorder=6,
                          label=f"Enemy {enemy_index}: every {interval} step{'s' if interval > 1 else ''}")
        label = ax.text(0, 0, f"E{enemy_index}", ha="center", va="center",
                        fontsize=6.5, color="white", weight="bold", zorder=8)
        enemy_markers.append(marker)
        enemy_labels.append(label)

    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7, framealpha=0.92)

    def update(frame_index: int):
        frame = frames[frame_index]
        ar, ac = frame["agent_position"]
        gr, gc = frame["goal_position"]
        enemy_positions = frame["moving_obstacle_positions"]
        paused = frame.get("enemy_paused_this_step", [False] * len(enemy_markers))
        pause_remaining = frame.get("enemy_pause_remaining", [0] * len(enemy_markers))

        agent_marker.set_offsets(np.asarray([[ac, ar]], dtype=float))
        goal_marker.set_offsets(np.asarray([[gc, gr]], dtype=float))
        agent_rows = [int(item["agent_position"][0]) for item in frames[: frame_index + 1]]
        agent_cols = [int(item["agent_position"][1]) for item in frames[: frame_index + 1]]
        goal_rows = [int(item["goal_position"][0]) for item in frames[: frame_index + 1]]
        goal_cols = [int(item["goal_position"][1]) for item in frames[: frame_index + 1]]
        agent_trail.set_data(agent_cols, agent_rows)
        goal_trail.set_data(goal_cols, goal_rows)

        for enemy_index, (position, marker, label) in enumerate(
            zip(enemy_positions, enemy_markers, enemy_labels)
        ):
            row, col = position
            is_paused = bool(paused[enemy_index]) or int(pause_remaining[enemy_index]) > 0
            marker.set_data([col], [row])
            marker.set_marker("s" if is_paused else "o")
            marker.set_markersize(11 if is_paused else 9)
            label.set_position((col, row))

        action_name = frame.get("action_name") or "START"
        reward = float(frame.get("reward", 0.0))
        episode_return = float(frame.get("episode_return", 0.0))
        paused_count = sum(bool(value) for value in paused)
        status = "SUCCESS" if frame.get("success") else "TIME LIMIT" if frame.get("truncated") else "RUNNING"
        if frame_index == len(frames) - 1:
            status = "SUCCESS" if payload.get("success") else (payload.get("final_reason") or "END").upper()
        ax.set_title(
            f"Training Episode {episode} | Step {frame.get('step', frame_index)} | "
            f"Action: {action_name} | Reward: {reward:+.2f} | Return: {episode_return:+.2f}\n"
            f"Paused enemies: {paused_count}/{len(enemy_markers)} | {status}",
            fontsize=10,
        )
        return [agent_marker, goal_marker, agent_trail, goal_trail, *enemy_markers, *enemy_labels]

    animation = FuncAnimation(
        fig, update, frames=len(frames), interval=1000 / fps,
        blit=False, repeat=False,
    )

    try:
        animation.save(gif_path, writer=PillowWriter(fps=fps))
        mp4_result: str | None = None
        if save_mp4 and shutil.which("ffmpeg"):
            try:
                animation.save(
                    mp4_path,
                    writer=FFMpegWriter(fps=fps, codec="h264", bitrate=1800),
                )
                mp4_result = str(mp4_path)
            except Exception:
                if mp4_path.exists():
                    mp4_path.unlink()
        else:
            mp4_result = None
    finally:
        plt.close(fig)

    return {
        "episode": episode,
        "success": bool(payload.get("success", False)),
        "steps": int(payload.get("episode_length", len(frames) - 1)),
        "gif_path": str(gif_path),
        "mp4_path": mp4_result,
    }


def render_stochastic_enemy_episode_videos(
    episode_dir: str | Path,
    episodes: Iterable[int],
    output_dir: str | Path | None = None,
    fps: int = 10,
    save_mp4: bool = True,
) -> list[dict[str, str | int | bool | None]]:
    """Render selected one-based recorded training episodes in numeric order."""
    episode_dir = Path(episode_dir)
    selected = sorted({int(value) for value in episodes})
    if not selected or any(value <= 0 for value in selected):
        raise ValueError("episodes must contain positive one-based episode numbers")
    missing = [
        value for value in selected
        if not (episode_dir / f"episode_{value:06d}.json").is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Recorded data is missing for episode(s): "
            + ", ".join(map(str, missing))
            + ". Set record_episodes during training and rerun the training cell."
        )
    return [
        render_stochastic_enemy_episode(
            episode_dir / f"episode_{episode:06d}.json",
            output_dir=output_dir,
            fps=fps,
            save_mp4=save_mp4,
        )
        for episode in selected
    ]
