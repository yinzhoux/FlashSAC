import argparse
import math
import os
import random
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, MutableMapping

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

import hydra
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
from matplotlib.collections import LineCollection
from matplotlib.patches import Ellipse
from omegaconf import DictConfig, OmegaConf, open_dict

from flash_rl.agents import create_agent
from flash_rl.envs import create_vec_env
from flash_rl.types import Tensor


_ENV_TO_NORMALIZER_PRESET: dict[str, str] = {
    "ant-v4": "ant_preset",
    "ant": "ant_preset",
    "halfcheetah-v4": "half_cheetah_preset",
    "half_cheetah-v4": "half_cheetah_preset",
    "half_cheetah": "half_cheetah_preset",
}

DEFAULT_NUM_RANDOM_TRAJECTORIES = 48
DEFAULT_NUM_VIDEO_REPEATS = 2
DEFAULT_VIDEO_FPS = 15
DEFAULT_VIDEO_SKIP_FRAMES = 1


def _resolve_obs_normalizer_type(cfg: DictConfig) -> None:
    raw_normalizer = getattr(cfg.agent, "obs_normalizer_type", "off")

    if raw_normalizer in (None, False, "off"):
        resolved = "off"
    elif raw_normalizer in (True, "preset"):
        env_name = str(cfg.env.env_name).strip().lower().replace(" ", "")
        if env_name not in _ENV_TO_NORMALIZER_PRESET:
            supported = ", ".join(sorted(_ENV_TO_NORMALIZER_PRESET.keys()))
            raise ValueError(
                "obs_normalizer_type='preset' requires a supported env preset. "
                f"Got env.env_name={cfg.env.env_name!r}. Supported keys: {supported}."
            )
        resolved = _ENV_TO_NORMALIZER_PRESET[env_name]
    elif isinstance(raw_normalizer, str):
        resolved = raw_normalizer
    else:
        raise ValueError(f"Unsupported obs_normalizer_type={raw_normalizer!r}")

    with open_dict(cfg):
        cfg.agent.obs_normalizer_type = resolved


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")


def _extract_frames(render_output: Any, num_envs: int) -> list[np.ndarray]:
    if render_output is None:
        return []

    if isinstance(render_output, np.ndarray):
        if render_output.ndim == 4 and render_output.shape[0] == num_envs:
            return [np.asarray(render_output[i]) for i in range(num_envs)]
        if render_output.ndim == 3:
            return [np.asarray(render_output)]

    if isinstance(render_output, (list, tuple)) and len(render_output) == num_envs:
        return [np.asarray(frame) for frame in render_output]

    return []


def _to_uint8(frame: np.ndarray) -> np.ndarray:
    if frame.dtype == np.uint8:
        return frame
    if np.issubdtype(frame.dtype, np.floating):
        frame = np.clip(frame, 0.0, 255.0)
    return frame.astype(np.uint8)


def _unwrap_single_env(vec_env: Any) -> Any:
    env = vec_env.envs[0] if hasattr(vec_env, "envs") else vec_env
    while hasattr(env, "env"):
        env = env.env
    return env


def _extract_coordinates(env: Any, env_name: str, observation: np.ndarray) -> np.ndarray:
    base_env = _unwrap_single_env(env)
    unwrapped = getattr(base_env, "unwrapped", base_env)
    if hasattr(unwrapped, "data") and hasattr(unwrapped.data, "qpos"):
        qpos = np.asarray(unwrapped.data.qpos, dtype=np.float32).reshape(-1)
        lowered = env_name.lower()
        if qpos.size >= 2 and any(name in lowered for name in ["ant", "humanoid"]):
            return qpos[:2].copy()
        if qpos.size >= 1:
            return np.array([qpos[0], 0.0], dtype=np.float32)

    obs = np.asarray(observation, dtype=np.float32).reshape(-1)
    if obs.size >= 2:
        return obs[:2].copy()
    if obs.size == 1:
        return np.array([obs[0], 0.0], dtype=np.float32)
    return np.zeros(2, dtype=np.float32)


def _normalize_skills(skills: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(skills, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    return skills / norms


def _get_2d_colors(points: np.ndarray, min_point: tuple[float, float], max_point: tuple[float, float]) -> np.ndarray:
    min_arr = np.asarray(min_point, dtype=np.float32)
    max_arr = np.asarray(max_point, dtype=np.float32)
    colors = (points - min_arr) / (max_arr - min_arr)
    colors = np.hstack((colors, (2 - np.sum(colors, axis=1, keepdims=True)) / 2))
    colors = np.clip(colors, 0, 1)
    return np.c_[colors, np.full(len(colors), 0.8, dtype=np.float32)]


def _pca_project(values: np.ndarray, out_dim: int = 2) -> np.ndarray:
    centered = values - values.mean(axis=0, keepdims=True)
    if centered.shape[1] <= out_dim:
        return centered
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    return centered @ vh[:out_dim].T


def _get_option_colors(options: np.ndarray, color_range: float = 4.0) -> np.ndarray:
    num_options, dim_option = options.shape

    if dim_option <= 2:
        if dim_option == 1:
            points = []
            scale = 2.0
            for option in options[:, 0]:
                if option < 0:
                    points.append((scale - (-option) * scale, scale))
                else:
                    points.append((scale, scale - option * scale))
            options = np.asarray(points, dtype=np.float32)
        return _get_2d_colors(options, (-color_range, -color_range), (color_range, color_range))

    if dim_option > 3:
        projected = _pca_project(options, out_dim=3)
    else:
        projected = options[:, :3]

    max_colors = np.full(3, color_range, dtype=np.float32)
    min_colors = np.full(3, -color_range, dtype=np.float32)
    projected = (projected - min_colors) / (max_colors - min_colors)
    projected = np.clip(projected, 0, 1)
    return np.c_[projected, np.full(len(projected), 0.8, dtype=np.float32)]


def _generate_random_options(skill_dim: int, num_random_trajectories: int, skill_type: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if skill_type == "discrete":
        eye_options = np.eye(skill_dim, dtype=np.float32)
        options = []
        labels = []
        for idx in range(skill_dim):
            num = num_random_trajectories // skill_dim + int(idx < (num_random_trajectories % skill_dim))
            for _ in range(num):
                options.append(eye_options[idx])
                labels.append(idx)
        options_arr = np.asarray(options, dtype=np.float32)
        cmap = cm.get_cmap("tab10" if skill_dim <= 10 else "tab20")
        colors = np.asarray([cmap(label % cmap.N) for label in labels], dtype=np.float32)
        return options_arr, colors

    rng = np.random.default_rng(seed)
    options = rng.standard_normal((num_random_trajectories, skill_dim), dtype=np.float32)
    options = _normalize_skills(options)
    return options, _get_option_colors(options * 4.0)


def _generate_video_options(skill_dim: int, skill_type: str, num_video_repeats: int, seed: int) -> np.ndarray:
    if skill_type == "discrete":
        return np.eye(skill_dim, dtype=np.float32).repeat(num_video_repeats, axis=0)

    if skill_dim == 2:
        radius = 1.0
        base = []
        for angle in [3, 2, 1, 4]:
            base.append([radius * math.cos(angle * math.pi / 4), radius * math.sin(angle * math.pi / 4)])
        base.append([0.0, 0.0])
        for angle in [0, 5, 6, 7]:
            base.append([radius * math.cos(angle * math.pi / 4), radius * math.sin(angle * math.pi / 4)])
        options = np.asarray(base, dtype=np.float32)
    else:
        rng = np.random.default_rng(seed + 1)
        options = rng.standard_normal((9, skill_dim), dtype=np.float32)
        options = _normalize_skills(options)
    return options.repeat(num_video_repeats, axis=0)


def _compute_phi_distribution(agent: Any, observations: np.ndarray, encoder_type: str) -> tuple[np.ndarray, np.ndarray | None]:
    obs_tensor = torch.as_tensor(observations, dtype=torch.float32, device=agent._device)
    with torch.no_grad():
        if encoder_type == "gaussian":
            encoder_net = agent._skill_encoder.network
            dist = encoder_net.encoder(obs_tensor)
            means = dist.mean.detach().cpu().numpy()
            stddevs = dist.stddev.detach().cpu().numpy()
            return means, stddevs
        features = agent._skill_encoder(observations=obs_tensor, training=False)
        return features.detach().cpu().numpy(), None


def _to_plot_2d(values: np.ndarray) -> np.ndarray:
    if values.shape[1] == 1:
        return np.concatenate([values, np.zeros((values.shape[0], 1), dtype=values.dtype)], axis=1)
    if values.shape[1] == 2:
        return values
    return _pca_project(values, out_dim=2)


def _stddevs_for_plot(stddevs: np.ndarray | None, means_2d: np.ndarray) -> np.ndarray | None:
    if stddevs is None:
        return None
    if stddevs.shape[1] == 1:
        pad = np.full((stddevs.shape[0], 1), 0.1, dtype=stddevs.dtype)
        return np.concatenate([stddevs, pad], axis=1)
    if stddevs.shape[1] == 2:
        return stddevs
    avg = stddevs.mean(axis=1, keepdims=True)
    return np.repeat(avg, 2, axis=1)


def _draw_2d_gaussians(
    means: np.ndarray,
    stddevs: np.ndarray,
    colors: np.ndarray,
    axis: plt.Axes,
    *,
    fill: bool = False,
    alpha: float = 0.8,
    adaptive_axis: bool = False,
    plot_axis: list[float] | None = None,
) -> None:
    square_axis_limit = 2.0
    unit_circle = Ellipse(xy=(0, 0), width=2, height=2, edgecolor="r", lw=1, facecolor="none", alpha=0.5)
    axis.add_patch(unit_circle)

    for mean, stddev, color in zip(means, stddevs, colors):
        ellipse = Ellipse(
            xy=mean,
            width=float(stddev[0] * 2),
            height=float(stddev[1] * 2),
            edgecolor=color,
            lw=1,
            facecolor=color if fill else "none",
            alpha=alpha,
        )
        axis.add_patch(ellipse)
        square_axis_limit = max(
            square_axis_limit,
            abs(float(mean[0] + stddev[0])),
            abs(float(mean[0] - stddev[0])),
            abs(float(mean[1] + stddev[1])),
            abs(float(mean[1] - stddev[1])),
        )

    axis.axis("scaled")
    if plot_axis is not None:
        axis.axis(plot_axis)
    elif adaptive_axis:
        limit = square_axis_limit * 1.2
        axis.set_xlim(-limit, limit)
        axis.set_ylim(-limit, limit)
    else:
        axis.set_xlim(-5, 5)
        axis.set_ylim(-5, 5)


def _plot_trajectory(axis: plt.Axes, coordinates: np.ndarray, color: np.ndarray) -> None:
    if len(coordinates) < 2:
        return
    linewidths = np.linspace(0.4, 1.6, len(coordinates) - 1)
    points = coordinates.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    collection = LineCollection(segments, linewidths=linewidths, color=color)
    axis.add_collection(collection)


def _prepare_video_grid(videos: list[list[np.ndarray]], skip_frames: int) -> np.ndarray:
    clipped = [frames[::skip_frames] for frames in videos if len(frames) > 0]
    if not clipped:
        raise ValueError("No frames were collected for video export.")

    max_length = max(len(frames) for frames in clipped)
    padded = []
    for frames in clipped:
        tail = frames[-1]
        if len(frames) < max_length:
            frames = frames + [tail] * (max_length - len(frames))
        padded.append(np.stack(frames, axis=0))

    video_tensor = np.stack(padded, axis=0)
    batch, steps, height, width, channels = video_tensor.shape

    if batch <= 3:
        n_cols = batch
    elif batch <= 9:
        n_cols = 3
    else:
        n_cols = 6
    if batch % n_cols != 0:
        pad_count = n_cols - (batch % n_cols)
        padding = np.zeros((pad_count, steps, height, width, channels), dtype=video_tensor.dtype)
        video_tensor = np.concatenate([video_tensor, padding], axis=0)
        batch = video_tensor.shape[0]
    n_rows = batch // n_cols

    video_tensor = video_tensor.reshape(n_rows, n_cols, steps, height, width, channels)
    video_tensor = video_tensor.transpose(2, 0, 3, 1, 4, 5)
    return video_tensor.reshape(steps, n_rows * height, n_cols * width, channels)


def _get_video_grid_shape(num_videos: int) -> tuple[int, int]:
    if num_videos <= 3:
        n_cols = num_videos
    elif num_videos <= 9:
        n_cols = 3
    else:
        n_cols = 6
    n_rows = math.ceil(num_videos / n_cols)
    return n_rows, n_cols


def _write_rollout_frame(frame_dir: Path, frame_index: int, frame: np.ndarray) -> None:
    frame_dir.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(frame_dir / f"{frame_index:05d}.png", _to_uint8(frame))


def _read_rollout_frame(frame_dir: Path, frame_index: int) -> np.ndarray:
    return np.asarray(imageio.imread(frame_dir / f"{frame_index:05d}.png"))


def _rollout_one_skill(
    agent: Any,
    env: Any,
    skill: np.ndarray,
    skill_type: str,
    env_name: str,
    max_episode_steps: int,
    frame_dir: Path | None = None,
) -> dict[str, Any]:
    agent.set_eval_skills(skill[None, :], normalize=skill_type == "continuous")

    observations, _ = env.reset()
    prev_transition: MutableMapping[str, Tensor] = {"next_observation": observations}

    first_observation = np.asarray(observations[0], dtype=np.float32).copy()
    last_observation = first_observation
    coordinates = [_extract_coordinates(env, env_name, first_observation)]
    frame_count = 0

    if frame_dir is not None:
        render_frames = _extract_frames(env.render(), num_envs=1)  # type: ignore[call-arg]
        if render_frames:
            _write_rollout_frame(frame_dir, frame_count, render_frames[0])
            frame_count += 1

    done = False
    steps = 0
    while not done and steps < max_episode_steps:
        actions = np.asarray(agent.sample_actions(interaction_step=0, prev_transition=prev_transition, training=False))
        next_observations, _, terminateds, truncateds, _ = env.step(actions)
        done = bool(terminateds[0] or truncateds[0])

        last_observation = np.asarray(next_observations[0], dtype=np.float32).copy()
        coordinates.append(_extract_coordinates(env, env_name, last_observation))
        prev_transition = {"next_observation": next_observations}

        if frame_dir is not None:
            render_frames = _extract_frames(env.render(), num_envs=1)  # type: ignore[call-arg]
            if render_frames:
                _write_rollout_frame(frame_dir, frame_count, render_frames[0])
                frame_count += 1

        steps += 1

    return {
        "skill": skill.copy(),
        "last_observation": last_observation,
        "coordinates": np.asarray(coordinates, dtype=np.float32),
        "frame_dir": frame_dir,
        "frame_count": frame_count,
    }


def _save_traj_plot(rollouts: list[dict[str, Any]], colors: np.ndarray, output_path: Path, plot_axis: list[float] | None) -> None:
    fig, axis = plt.subplots(figsize=(6, 6), dpi=180)
    for rollout, color in zip(rollouts, colors):
        _plot_trajectory(axis, rollout["coordinates"], color)
    axis.autoscale()
    axis.set_title("TrajPlot_RandomZ")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.axis("equal")
    if plot_axis is not None:
        axis.axis(plot_axis)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _save_phi_plot(
    rollouts: list[dict[str, Any]],
    colors: np.ndarray,
    output_path: Path,
    agent: Any,
    encoder_type: str,
    plot_axis: list[float] | None,
) -> None:
    last_obs = np.stack([rollout["last_observation"] for rollout in rollouts], axis=0)
    phi_means, phi_stddevs = _compute_phi_distribution(agent, last_obs, encoder_type=encoder_type)

    means_2d = _to_plot_2d(phi_means)
    stddevs_2d = _stddevs_for_plot(phi_stddevs, means_2d)

    fig, axis = plt.subplots(figsize=(6, 6), dpi=180)
    if stddevs_2d is not None:
        _draw_2d_gaussians(means_2d, stddevs_2d, colors, axis, adaptive_axis=False, plot_axis=plot_axis)
    axis.scatter(means_2d[:, 0], means_2d[:, 1], c=colors, s=28, alpha=0.95)
    if stddevs_2d is not None:
        point_std = np.full((means_2d.shape[0], 2), 0.03, dtype=np.float32)
        _draw_2d_gaussians(
            means_2d,
            point_std,
            colors,
            axis,
            fill=True,
            adaptive_axis=True,
            plot_axis=plot_axis,
        )
    else:
        if plot_axis is not None:
            axis.axis(plot_axis)
        else:
            axis.axis("equal")
    axis.set_title("PhiPlot")
    axis.set_xlabel("phi[0]")
    axis.set_ylabel("phi[1]")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _save_video(rollouts: list[dict[str, Any]], output_path: Path, fps: int, skip_frames: int) -> None:
    valid_rollouts = [rollout for rollout in rollouts if rollout["frame_dir"] is not None and rollout["frame_count"] > 0]
    if not valid_rollouts:
        raise ValueError("No frames were collected for video export.")

    sample_frame = _read_rollout_frame(valid_rollouts[0]["frame_dir"], 0)
    black_frame = np.zeros_like(sample_frame)
    max_frame_count = max(int(rollout["frame_count"]) for rollout in valid_rollouts)
    n_rows, n_cols = _get_video_grid_shape(len(valid_rollouts))

    writer = imageio.get_writer(output_path, fps=fps)
    try:
        for frame_index in range(0, max_frame_count, skip_frames):
            panel_frames: list[np.ndarray] = []
            for rollout in valid_rollouts:
                last_valid = int(rollout["frame_count"]) - 1
                source_index = min(frame_index, last_valid)
                panel_frames.append(_read_rollout_frame(rollout["frame_dir"], source_index))

            while len(panel_frames) < n_rows * n_cols:
                panel_frames.append(black_frame)

            rows = []
            for row_idx in range(n_rows):
                start = row_idx * n_cols
                end = start + n_cols
                rows.append(np.concatenate(panel_frames[start:end], axis=1))
            writer.append_data(np.concatenate(rows, axis=0))
    finally:
        writer.close()


def generate_eval_artifacts(
    agent: Any,
    env: Any,
    cfg: DictConfig,
    output_dir: str | Path,
    *,
    num_random_trajectories: int = DEFAULT_NUM_RANDOM_TRAJECTORIES,
    num_video_repeats: int = DEFAULT_NUM_VIDEO_REPEATS,
    video_fps: int = DEFAULT_VIDEO_FPS,
    video_skip_frames: int = DEFAULT_VIDEO_SKIP_FRAMES,
    eval_plot_axis: list[float] | None = None,
) -> dict[str, Path]:
    if cfg.agent.agent_type != "metra":
        raise ValueError("This helper only supports METRA checkpoints/agents.")
    if not hasattr(agent, "set_eval_skills"):
        raise ValueError("Loaded METRA agent does not expose set_eval_skills().")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    skill_dim = int(cfg.agent.skill_dim)
    skill_type = str(cfg.agent.skill_type)
    random_options, option_colors = _generate_random_options(
        skill_dim=skill_dim,
        num_random_trajectories=num_random_trajectories,
        skill_type=skill_type,
        seed=int(cfg.seed),
    )
    video_options = _generate_video_options(
        skill_dim=skill_dim,
        skill_type=skill_type,
        num_video_repeats=num_video_repeats,
        seed=int(cfg.seed),
    )

    rollout_max_steps = int(getattr(cfg, "max_path_length", cfg.env.max_episode_steps))
    traj_rollouts = [
        _rollout_one_skill(
            agent=agent,
            env=env,
            skill=skill,
            skill_type=skill_type,
            env_name=str(cfg.env.env_name),
            max_episode_steps=rollout_max_steps,
        )
        for skill in random_options
    ]

    phi_plot_path = output_path / "PhiPlot.png"
    traj_plot_path = output_path / "TrajPlot_RandomZ.png"
    video_path = output_path / "Video_RandomZ.mp4"

    with TemporaryDirectory(prefix="metra_eval_video_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        video_rollouts = [
            _rollout_one_skill(
                agent=agent,
                env=env,
                skill=skill,
                skill_type=skill_type,
                env_name=str(cfg.env.env_name),
                max_episode_steps=rollout_max_steps,
                frame_dir=tmp_dir / f"skill_{idx:02d}",
            )
            for idx, skill in enumerate(video_options)
        ]

        _save_traj_plot(traj_rollouts, option_colors, traj_plot_path, eval_plot_axis)
        _save_phi_plot(
            traj_rollouts,
            option_colors,
            phi_plot_path,
            agent,
            encoder_type=str(cfg.agent.encoder_type),
            plot_axis=eval_plot_axis,
        )
        _save_video(
            video_rollouts,
            video_path,
            fps=video_fps,
            skip_frames=video_skip_frames,
        )

    return {
        "phi_plot_path": phi_plot_path,
        "traj_plot_path": traj_plot_path,
        "video_path": video_path,
    }


def run(args: argparse.Namespace) -> None:
    OmegaConf.register_new_resolver("eval", lambda s: eval(s))
    hydra.initialize(version_base=None, config_path=args.config_path)
    cfg = hydra.compose(config_name=args.config_name, overrides=args.overrides)
    OmegaConf.resolve(cfg)
    _resolve_obs_normalizer_type(cfg)

    with open_dict(cfg):
        cfg.agent.load_optimizer = False
        cfg.agent.load_reward_normalizer = False
        cfg.env.num_train_envs = 1
        cfg.env.num_eval_envs = 1
        cfg.env.num_record_envs = 1

    _seed_everything(int(cfg.seed))

    record_env = None
    try:
        record_env = create_vec_env(
            env_type=cfg.env.env_type,
            env_name=cfg.env.env_name,
            num_envs=1,
            seed=int(cfg.seed),
            rescale_action=cfg.env.rescale_action,
            max_episode_steps=cfg.env.max_episode_steps,
        )
        observation_space = record_env.observation_space
        action_space = record_env.action_space
        _, env_info = record_env.reset()

        agent = create_agent(
            observation_space=observation_space,
            action_space=action_space,
            env_info=env_info,
            cfg=cfg.agent,
        )
        agent.load(args.checkpoint_path)

        timestamp = datetime.now().strftime("%m%d-%H%M%S")
        run_name = args.run_name or f"{Path(args.checkpoint_path).name}_{timestamp}"
        output_dir = Path(args.output_dir) / run_name
        plot_axis = list(args.eval_plot_axis) if args.eval_plot_axis else None
        generate_eval_artifacts(
            agent=agent,
            env=record_env,
            cfg=cfg,
            output_dir=output_dir,
            num_random_trajectories=args.num_random_trajectories,
            num_video_repeats=args.num_video_repeats,
            video_fps=args.video_fps,
            video_skip_frames=args.video_skip_frames,
            eval_plot_axis=plot_axis,
        )

        print(f"Saved evaluation artifacts to: {output_dir}")
    finally:
        if record_env is not None:
            record_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate official-style METRA evaluation artifacts from a FlashSAC checkpoint.")
    parser.add_argument("--checkpoint-path", type=str, required=True, help="Path to the saved FlashSAC METRA checkpoint directory.")
    parser.add_argument("--config-path", type=str, default="configs")
    parser.add_argument("--config-name", type=str, default="metra_aligned_base")
    parser.add_argument("--overrides", action="append", default=[])
    parser.add_argument("--output-dir", type=str, default="./eval_artifacts")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--num-random-trajectories", type=int, default=48)
    parser.add_argument("--num-video-repeats", type=int, default=2)
    parser.add_argument("--video-fps", type=int, default=15)
    parser.add_argument("--video-skip-frames", type=int, default=1)
    parser.add_argument("--eval-plot-axis", type=float, nargs="*", default=None)
    run(parser.parse_args())