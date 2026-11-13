"""Render recorded training episodes as animated GIF and optional MP4 files.

Episode records are captured by ``navigation_rl.dqn.train_agent`` during the
actual training interaction. This module only visualizes those saved states; it
does not rerun the policy or reconstruct an approximate trajectory.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap


def _load_episode_record(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported episode record schema in {path}")
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"Episode record has no frames: {path}")
    required = {"agent_position", "goal_position", "moving_obstacle_positions", "step"}
    for index, frame in enumerate(frames):
        missing = required.difference(frame)
        if missing:
            raise ValueError(f"Frame {index} in {path} is missing: {sorted(missing)}")
    return payload


def render_episode_video(
    record_path: str | Path,
    output_dir: str | Path | None = None,
    fps: int = 10,
    save_mp4: bool = True,
) -> dict[str, str | int | bool | None]:
    """Render one recorded episode to GIF and, when possible, MP4."""
    record_path = Path(record_path)
    if fps <= 0:
        raise ValueError("fps must be a positive integer")
    if not record_path.is_file():
        raise FileNotFoundError(record_path)

    payload = _load_episode_record(record_path)
    metadata = payload["environment"]
    frames = payload["frames"]
    grid_size = int(metadata["grid_size"])
    walls = {tuple(position) for position in metadata["static_walls"]}
    episode = int(payload["episode"])

    output_dir = Path(output_dir) if output_dir is not None else record_path.parent / "videos"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"episode_{episode:06d}"
    gif_path = output_dir / f"{stem}.gif"
    mp4_path = output_dir / f"{stem}.mp4"

    background = np.zeros((grid_size, grid_size), dtype=np.uint8)
    for row, col in walls:
        background[row, col] = 1

    fig, ax = plt.subplots(figsize=(8, 8))
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

    agent_marker = ax.scatter([], [], s=155, color="#2878b5", marker="o", label="Agent", zorder=6)
    goal_marker = ax.scatter([], [], s=250, color="#2b9a62", marker="*", label="Goal", zorder=6)
    obstacle_marker = ax.scatter([], [], s=120, color="#cf4545", marker="X", label="Moving obstacles", zorder=5)
    agent_trail, = ax.plot([], [], color="#2878b5", linewidth=2.0, alpha=0.8, label="Agent path", zorder=4)
    goal_trail, = ax.plot([], [], color="#2b9a62", linewidth=1.4, linestyle="--", alpha=0.7, label="Goal path", zorder=3)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.92)

    def update(frame_index: int):
        frame = frames[frame_index]
        ar, ac = frame["agent_position"]
        gr, gc = frame["goal_position"]
        obstacles = frame["moving_obstacle_positions"]
        agent_rows = [int(item["agent_position"][0]) for item in frames[: frame_index + 1]]
        agent_cols = [int(item["agent_position"][1]) for item in frames[: frame_index + 1]]
        goal_rows = [int(item["goal_position"][0]) for item in frames[: frame_index + 1]]
        goal_cols = [int(item["goal_position"][1]) for item in frames[: frame_index + 1]]

        agent_marker.set_offsets(np.asarray([[ac, ar]], dtype=float))
        goal_marker.set_offsets(np.asarray([[gc, gr]], dtype=float))
        obstacle_offsets = np.asarray([[position[1], position[0]] for position in obstacles], dtype=float)
        if obstacle_offsets.size == 0:
            obstacle_offsets = np.empty((0, 2), dtype=float)
        obstacle_marker.set_offsets(obstacle_offsets)
        agent_trail.set_data(agent_cols, agent_rows)
        goal_trail.set_data(goal_cols, goal_rows)

        action_name = frame.get("action_name") or "START"
        reward = float(frame.get("reward", 0.0))
        episode_return = float(frame.get("episode_return", 0.0))
        status = "SUCCESS" if frame.get("success") else "TIME LIMIT" if frame.get("truncated") else "RUNNING"
        if frame_index == len(frames) - 1:
            status = "SUCCESS" if payload.get("success") else (payload.get("final_reason") or "END").upper()
        ax.set_title(
            f"Training Episode {episode} | Step {frame.get('step', frame_index)} | "
            f"Action: {action_name} | Reward: {reward:+.2f} | "
            f"Return: {episode_return:+.2f}\n{status}",
            fontsize=10,
        )
        return agent_marker, goal_marker, obstacle_marker, agent_trail, goal_trail

    animation = FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=1000 / fps,
        blit=False,
        repeat=False,
    )

    try:
        animation.save(gif_path, writer=PillowWriter(fps=fps))
        mp4_result: str | None = None
        if save_mp4:
            ffmpeg_path = shutil.which("ffmpeg")
            if ffmpeg_path:
                try:
                    animation.save(
                        mp4_path,
                        writer=FFMpegWriter(fps=fps, codec="h264", bitrate=1800),
                    )
                    mp4_result = str(mp4_path)
                except Exception:
                    mp4_result = None
                    if mp4_path.exists():
                        mp4_path.unlink()
    finally:
        plt.close(fig)

    return {
        "episode": episode,
        "success": bool(payload.get("success", False)),
        "steps": int(payload.get("episode_length", len(frames) - 1)),
        "gif_path": str(gif_path),
        "mp4_path": mp4_result,
    }


def render_episode_videos(
    episode_dir: str | Path,
    episodes: Iterable[int],
    output_dir: str | Path | None = None,
    fps: int = 10,
    save_mp4: bool = True,
) -> list[dict[str, str | int | bool | None]]:
    """Render a selected set of recorded training episodes in numeric order."""
    episode_dir = Path(episode_dir)
    selected = sorted({int(value) for value in episodes})
    if not selected:
        raise ValueError("episodes must contain at least one episode number")
    if any(value <= 0 for value in selected):
        raise ValueError("episode numbers must be positive and one-based")

    missing = [episode for episode in selected if not (episode_dir / f"episode_{episode:06d}.json").is_file()]
    if missing:
        expected = ", ".join(str(value) for value in missing)
        raise FileNotFoundError(
            f"Recorded training data is missing for episode(s): {expected}. "
            f"Run the training cell with record_episodes={selected} first. "
            "Past episodes cannot be reconstructed unless their trajectories were recorded."
        )

    results = []
    for episode in selected:
        record_path = episode_dir / f"episode_{episode:06d}.json"
        results.append(
            render_episode_video(
                record_path=record_path,
                output_dir=output_dir,
                fps=fps,
                save_mp4=save_mp4,
            )
        )
    return results
