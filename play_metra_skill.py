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


    # 自动生成10个skill方向（单位圆等分）
    skill_dim = int(cfg.agent.skill_dim)
    assert skill_dim == 2, "只支持 skill_dim=2 的情况（二维skill）"
    num_skills = 10
    angles = np.linspace(0, 2 * np.pi, num_skills, endpoint=False)
    skills_list = np.stack([np.cos(angles), np.sin(angles)], axis=1)  # (10, 2)

    for idx, skill in enumerate(skills_list):
        agent.set_eval_skills(skill[None, :], normalize=True)

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

        initial_frames = _extract_frames(env.render(), num_envs=1)
        if initial_frames and initial_frames[0].size > 0:
            episode_frames.append(_to_uint8(initial_frames[0]))

        done = False
        episode_return = 0.0
        episode_length = 0
        while not done:
            actions = agent.sample_actions(interaction_step=0, prev_transition=prev_transition, training=False)
            actions = np.asarray(actions)
            next_observations, rewards, terminateds, truncateds, _ = env.step(actions)
            dones = np.logical_or(terminateds, truncateds)

            step_frames = _extract_frames(env.render(), num_envs=1)
            if step_frames and step_frames[0].size > 0:
                episode_frames.append(_to_uint8(step_frames[0]))

            episode_actions.append(np.asarray(actions[0]).copy())
            episode_rewards.append(float(rewards[0]))
            episode_terminated.append(bool(terminateds[0]))
            episode_truncated.append(bool(truncateds[0]))
            episode_obs.append(np.asarray(next_observations[0]).copy())

            episode_return += float(rewards[0])
            episode_length += 1
            done = bool(dones[0])
            prev_transition = {"next_observation": next_observations}

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
    # skill参数已废弃
    # parser.add_argument(
    #     "--skill",
    #     type=float,
    #     nargs="+",
    #     required=True,
    #     help="Skill values. Provide skill_dim values for broadcast, or num_envs*skill_dim for per-env skills.",
    # )
    # parser.add_argument("--no_normalize_skill", action="store_true")
    parser.add_argument("--output_dir", type=str, default="./play_records")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--video_fps", type=int, default=30)
    args = parser.parse_args()
    play_and_record(args)
