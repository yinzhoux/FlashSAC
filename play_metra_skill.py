import argparse
import random
from datetime import datetime
from pathlib import Path
from typing import Any, MutableMapping

import hydra
import imageio.v2 as imageio
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont

from flash_rl.agents import create_agent
from flash_rl.envs import create_vec_env
from flash_rl.types import Tensor


def _parse_skill_values(values: list[float], num_envs: int, skill_dim: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == skill_dim:
        return np.repeat(arr[None, :], repeats=num_envs, axis=0)
    if arr.size == num_envs * skill_dim:
        return arr.reshape(num_envs, skill_dim)
    raise ValueError(
        f"Invalid --skill length {arr.size}. Expected {skill_dim} (broadcast to all envs) "
        f"or {num_envs * skill_dim} (per-env skills)."
    )


def _extract_frames(render_output: Any, num_envs: int) -> list[np.ndarray]:
    if render_output is None:
        return []

    if isinstance(render_output, np.ndarray):
        if render_output.ndim == 4 and render_output.shape[0] == num_envs:
            return [render_output[i] for i in range(num_envs)]
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


def _format_vector(name: str, values: np.ndarray, precision: int = 3) -> str:
    formatted = ", ".join(f"{float(v):.{precision}f}" for v in values)
    return f"{name}: [{formatted}]"


def _compute_phi(agent: Any, observation: np.ndarray) -> np.ndarray:
    obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=agent._device).unsqueeze(0)
    with torch.no_grad():
        phi = agent._skill_encoder(observations=obs_tensor, training=False)
    return phi.squeeze(0).detach().cpu().numpy()


def _compute_intrinsic_reward(current_phi: np.ndarray, next_phi: np.ndarray, skill: np.ndarray, reward_scale: float) -> float:
    delta_phi = next_phi - current_phi
    alignment = float(np.dot(delta_phi, skill))
    return reward_scale * alignment


def _compute_discrete_skill_mask(skill: np.ndarray) -> np.ndarray:
    skill_dim = skill.shape[-1]
    if skill_dim <= 1:
        raise ValueError(f"Discrete skill masks require skill_dim > 1, got {skill_dim}")
    mean_skill = np.mean(skill, keepdims=True)
    return (skill - mean_skill) * (skill_dim / (skill_dim - 1))


def _format_skill(skill: np.ndarray, skill_type: str) -> str:
    if skill_type == "discrete":
        return f"skill_index: {int(np.argmax(skill))}"
    return _format_vector("skill", skill)


def _overlay_text(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    image = Image.fromarray(_to_uint8(frame))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    left = 10
    top = 10
    line_height = 14
    box_height = 10 + line_height * len(lines)
    box_width = min(image.width - 20, max(220, max(len(line) for line in lines) * 7))
    draw.rectangle((left - 4, top - 4, left + box_width, top + box_height), fill=(0, 0, 0, 160))

    for idx, line in enumerate(lines):
        draw.text((left, top + idx * line_height), line, fill=(255, 255, 255), font=font)

    return np.asarray(image)


def _generate_eval_skills(skill_dim: int, num_skills: int, seed: int, skill_type: str) -> np.ndarray:
    if skill_type == "discrete":
        return np.eye(skill_dim, dtype=np.float32)

    if skill_dim == 2:
        angles = np.linspace(0, 2 * np.pi, num_skills, endpoint=False, dtype=np.float32)
        return np.stack([np.cos(angles), np.sin(angles)], axis=1)

    rng = np.random.default_rng(seed)
    skills = rng.standard_normal((num_skills, skill_dim), dtype=np.float32)
    norms = np.linalg.norm(skills, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    return skills / norms


def play_and_record(args: argparse.Namespace) -> None:
    OmegaConf.register_new_resolver("eval", lambda s: eval(s))
    hydra.initialize(version_base=None, config_path=args.config_path)
    cfg = hydra.compose(config_name=args.config_name, overrides=args.overrides)
    OmegaConf.resolve(cfg)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    if cfg.env.env_type == "isaaclab":
        raise ValueError("IsaacLab wrapper in this repo does not support env.render(), so mp4 export is unavailable.")

    num_envs = int(args.num_envs)
    env = create_vec_env(
        env_type=cfg.env.env_type,
        env_name=cfg.env.env_name,
        num_envs=num_envs,
        seed=cfg.seed,
        rescale_action=cfg.env.rescale_action,
        max_episode_steps=cfg.env.max_episode_steps,
    )

    observations, env_info = env.reset()
    agent = create_agent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        env_info=env_info,
        cfg=cfg.agent,
    )
    agent.load(args.checkpoint_path)

    if cfg.agent.agent_type != "metra":
        env.close()
        raise ValueError("This script is for METRA agent checkpoints only.")
    if not hasattr(agent, "set_eval_skills"):
        env.close()
        raise ValueError("METRA agent does not expose set_eval_skills().")


    # 2D skill: unit circle sweep. Higher-D skill: deterministic unit hypersphere samples.
    skill_dim = int(cfg.agent.skill_dim)
    skill_type = str(getattr(cfg.agent, "skill_type", "continuous"))
    num_skills = skill_dim if skill_type == "discrete" else 10
    skills_list = _generate_eval_skills(skill_dim=skill_dim, num_skills=num_skills, seed=cfg.seed, skill_type=skill_type)

    for idx, skill in enumerate(skills_list):
        agent.set_eval_skills(skill[None, :], normalize=skill_type == "continuous")

        run_name = args.run_name or f"{cfg.env.env_name}_seed{cfg.seed}_{datetime.now().strftime('%m%d-%H%M%S')}_skill{idx:02d}"
        output_root = Path(args.output_dir) / run_name
        traj_dir = output_root / "trajectories"
        video_dir = output_root / "videos"
        traj_dir.mkdir(parents=True, exist_ok=True)
        video_dir.mkdir(parents=True, exist_ok=True)

        print(f"[Info] Output dir: {output_root}")
        print(f"[Info] Skill {idx}: {skill}")

        # 重置环境
        observations, env_info = env.reset()
        prev_transition: MutableMapping[str, Tensor] = {"next_observation": observations}
        episode_obs = [np.asarray(observations[0]).copy()]
        episode_actions = []
        episode_rewards = []
        episode_terminated = []
        episode_truncated = []
        episode_frames = []

        current_phi = _compute_phi(agent, np.asarray(observations[0]))

        initial_frames = _extract_frames(env.render(), num_envs=1)
        if initial_frames and initial_frames[0].size > 0:
            episode_frames.append(
                _overlay_text(
                    initial_frames[0],
                    [
                        "step: 0",
                        _format_skill(skill, skill_type),
                        _format_vector("phi", current_phi),
                        "intrinsic_reward: 0.000",
                    ],
                )
            )

        done = False
        episode_return = 0.0
        episode_length = 0
        while not done:
            actions = agent.sample_actions(interaction_step=0, prev_transition=prev_transition, training=False)
            actions = np.asarray(actions)
            next_observations, rewards, terminateds, truncateds, _ = env.step(actions)
            dones = np.logical_or(terminateds, truncateds)
            next_phi = _compute_phi(agent, np.asarray(next_observations[0]))
            intrinsic_reward = _compute_intrinsic_reward(
                current_phi=current_phi,
                next_phi=next_phi,
                skill=_compute_discrete_skill_mask(skill) if skill_type == "discrete" else skill,
                reward_scale=float(getattr(cfg.agent, "skill_reward_scale", 1.0)),
            )

            step_frames = _extract_frames(env.render(), num_envs=1)
            if step_frames and step_frames[0].size > 0:
                episode_frames.append(
                    _overlay_text(
                        step_frames[0],
                        [
                            f"step: {episode_length + 1}",
                            _format_skill(skill, skill_type),
                            _format_vector("phi", next_phi),
                            f"intrinsic_reward: {intrinsic_reward:.3f}",
                        ],
                    )
                )

            episode_actions.append(np.asarray(actions[0]).copy())
            episode_rewards.append(float(rewards[0]))
            episode_terminated.append(bool(terminateds[0]))
            episode_truncated.append(bool(truncateds[0]))
            episode_obs.append(np.asarray(next_observations[0]).copy())

            episode_return += float(rewards[0])
            episode_length += 1
            done = bool(dones[0])
            prev_transition = {"next_observation": next_observations}
            current_phi = next_phi

        # 保存
        traj_path = traj_dir / f"episode_skill{idx:02d}.npz"
        np.savez_compressed(
            traj_path,
            observations=np.asarray(episode_obs, dtype=np.float32),
            actions=np.asarray(episode_actions, dtype=np.float32),
            rewards=np.asarray(episode_rewards, dtype=np.float32),
            terminated=np.asarray(episode_terminated, dtype=bool),
            truncated=np.asarray(episode_truncated, dtype=bool),
            skill=skill.astype(np.float32),
            episode_return=np.asarray(episode_return, dtype=np.float32),
            episode_length=np.asarray(episode_length, dtype=np.int32),
        )

        video_path = video_dir / f"episode_skill{idx:02d}.mp4"
        if len(episode_frames) > 0:
            imageio.mimwrite(video_path, episode_frames, fps=args.video_fps)

        print(f"[Done] Skill {idx} | Return: {episode_return:.2f} | Length: {episode_length}")

    env.close()
    print(f"[All Done] Saved all skill rollouts to: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Play METRA checkpoint with user-provided skill and record trajectories/videos."
    )
    parser.add_argument("--config_path", type=str, default="./configs")
    parser.add_argument("--config_name", type=str, default="metra_ant_pretrain_base")
    parser.add_argument("--overrides", action="append", default=[])
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="./play_records")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--video_fps", type=int, default=30)
    args = parser.parse_args()
    play_and_record(args)
