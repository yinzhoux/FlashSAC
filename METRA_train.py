"""METRA training script aligned with the official training rhythm.

Official rhythm (from ``METRA/tests/main.py``):
  for epoch in ...:
    1. Collect ``traj_batch_size`` complete episodes with FROZEN networks.
    2. Perform ``trans_optimization_epochs`` gradient updates (batch=trans_minibatch_size).

This contrasts with ``train.py`` which does 1 gradient update per interaction step
(online / step-by-step), resulting in ~4× more updates per env-step.

Usage example (Ant-v4, matching official command defaults):
  python METRA_train.py \\
      --config-path configs --config-name metra_aligned_base \\
      env.env_name=Ant-v4 \\
      group_name=debug exp_name=ant_aligned seed=0
"""

import os

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

import argparse
import random
import shutil
import sys
from datetime import datetime
from typing import Optional

import hydra
import numpy as np
import torch
import tqdm
from omegaconf import OmegaConf

from flash_rl.agents import create_agent
from flash_rl.common import create_logger
from flash_rl.envs import create_envs
from flash_rl.evaluation import evaluate, record_video
from flash_rl.types import Tensor


# ---------------------------------------------------------------------------
# Metric-name mapping: FlashSAC native keys → official METRA keys
#
# The official implementation logs under the prefix "METRA/" (from name='METRA'
# in tests/main.py).  We reproduce those names here so both TensorBoard runs
# can be compared side-by-side without renaming.
#
# Official source files:
#   iod/metra.py  → PureRewardMean, TeObjMean, LossTe, DualCstPenalty
#   iod/iod.py    → DualLam, LossDualLam, PathLengthMean/Max/Min
#   iod/sac_utils.py → LossQf1, LossQf2, SacpNewActionLogProbMean, LossSacp, Alpha, LossAlpha
# ---------------------------------------------------------------------------
_METRA_KEY_MAP: dict[str, str] = {
    # Encoder / METRA objective
    "skill_encoder/mean_intrinsic_reward": "METRA/PureRewardMean",
    "skill_encoder/mean_objective":        "METRA/TeObjMean",
    "skill_encoder/mean_constraint":       "METRA/DualCstPenalty",
    # Dual lambda
    "dual_lambda/value":                   "METRA/DualLam",
    "dual_lambda/loss":                    "METRA/LossDualLam",
    # SAC components
    "actor/loss":                          "METRA/LossSacp",
    "critic/loss":                         "METRA/LossQf",   # C51 CE loss ≈ LossQf1+LossQf2
    "temperature/value":                   "METRA/Alpha",
    "temperature/loss":                    "METRA/LossAlpha",
}


def _remap_to_metra_keys(update_info: dict) -> dict:
    """Return a new dict with official METRA metric names for easy side-by-side
    TensorBoard comparison.

    Derived quantities:
      METRA/LossTe                  = -METRA/TeObjMean    (official: loss = -objective)
      METRA/SacpNewActionLogProbMean = -actor/entropy      (log_prob = -entropy)
    """
    metra: dict = {}
    for flashsac_key, metra_key in _METRA_KEY_MAP.items():
        if flashsac_key in update_info:
            metra[metra_key] = update_info[flashsac_key]
    # Derived
    if "skill_encoder/mean_objective" in update_info:
        metra["METRA/LossTe"] = -update_info["skill_encoder/mean_objective"]
    if "actor/entropy" in update_info:
        metra["METRA/SacpNewActionLogProbMean"] = -update_info["actor/entropy"]
    return metra


def _prune_old_checkpoints(save_path_base: str, max_checkpoints_to_keep: Optional[int]) -> None:
    if max_checkpoints_to_keep is None or max_checkpoints_to_keep <= 0:
        return

    checkpoint_dirs: list[tuple[int, str]] = []
    if not os.path.isdir(save_path_base):
        return

    for name in os.listdir(save_path_base):
        if not name.startswith("epoch"):
            continue
        step_str = name[5:]
        if not step_str.isdigit():
            continue
        checkpoint_dirs.append((int(step_str), os.path.join(save_path_base, name)))

    checkpoint_dirs.sort(key=lambda item: item[0], reverse=True)
    for _, path in checkpoint_dirs[max_checkpoints_to_keep:]:
        shutil.rmtree(path, ignore_errors=True)


def run(args: argparse.Namespace) -> None:
    # ─────────────────────────────────────────────────────────────────────────
    # Config
    # ─────────────────────────────────────────────────────────────────────────
    OmegaConf.register_new_resolver("eval", lambda s: eval(s))

    hydra.initialize(version_base=None, config_path=args.config_path)
    cfg = hydra.compose(config_name=args.config_name, overrides=args.overrides)
    OmegaConf.resolve(cfg)

    assert cfg.num_train_envs == 1, (
        "METRA_train.py uses sequential episode collection and requires num_train_envs=1. "
        f"Got num_train_envs={cfg.num_train_envs}."
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Seeding
    # ─────────────────────────────────────────────────────────────────────────
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    # ─────────────────────────────────────────────────────────────────────────
    # Environments
    # ─────────────────────────────────────────────────────────────────────────
    train_env, eval_env, record_env = create_envs(**cfg.env)

    observation_space = train_env.observation_space
    action_space = train_env.action_space

    # ─────────────────────────────────────────────────────────────────────────
    # Agent
    # ─────────────────────────────────────────────────────────────────────────
    _, env_info = train_env.reset()
    agent = create_agent(
        observation_space=observation_space,
        action_space=action_space,
        env_info=env_info,
        cfg=cfg.agent,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Logger / checkpoint paths
    # ─────────────────────────────────────────────────────────────────────────
    logger = create_logger(cfg)

    script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    save_path_resolved = cfg.save_path.replace("TIMESTAMP", datetime.now().strftime("%m%d-%H%M%S"))
    save_path_base = os.path.join(script_dir, save_path_resolved)

    if cfg.agent_load_path is not None:
        agent.load(os.path.join(script_dir, cfg.agent_load_path))
    if cfg.buffer_load_path is not None:
        agent.load_replay_buffer(os.path.join(script_dir, cfg.buffer_load_path))

    # ─────────────────────────────────────────────────────────────────────────
    # Initial evaluation
    # ─────────────────────────────────────────────────────────────────────────
    eval_info = evaluate(agent, eval_env, cfg.num_eval_episodes, cfg.env.env_type)
    video_info = record_video(agent, record_env, cfg.num_record_episodes, cfg.env.env_type)
    logger.update_metric(**eval_info)
    logger.update_metric(**video_info)
    logger.log_metric(step=0)
    logger.reset()

    # ─────────────────────────────────────────────────────────────────────────
    # Training loop  (epoch-based, aligned with official METRA)
    # ─────────────────────────────────────────────────────────────────────────
    total_env_steps = 0
    update_info: dict = {}

    for epoch in tqdm.tqdm(range(cfg.n_epochs), smoothing=0.1, mininterval=0.5):

        # ── Phase 1: Collect traj_batch_size complete episodes ────────────────
        # Networks are not updated during collection (frozen weights).
        episode_returns: list[float] = []
        episode_lengths: list[int] = []

        for _episode_idx in range(cfg.traj_batch_size):
            observations, env_infos_reset = train_env.reset()
            prev_transition = {"next_observation": observations}
            episode_return = 0.0
            episode_length = 0

            for step in range(cfg.max_path_length):
                # Use random actions during warm-up (before buffer is full enough)
                if agent.can_start_training():
                    actions = agent.sample_actions(step, prev_transition, training=True)
                else:
                    actions = train_env.action_space.sample()

                actions = np.array(actions)
                next_observations, rewards, terminateds, truncateds, env_infos = train_env.step(actions)

                # ── Handle final observation ──────────────────────────────────
                # In gymnasium VectorEnv, on episode end the env auto-resets and
                # returns the NEW episode's first obs in next_observations.
                # The actual terminal observation is stored in info["final_obs"].
                next_buffer_observations = next_observations.copy()
                naturally_done = bool(terminateds[0]) or bool(truncateds[0])
                if naturally_done:
                    next_buffer_observations[0] = env_infos["final_obs"][0]

                # ── Force-truncate at max_path_length boundary ────────────────
                # Matches official METRA which truncates trajectories at
                # max_path_length steps even if the env has a longer time limit.
                at_path_limit = (step == cfg.max_path_length - 1)
                effective_truncateds = truncateds.copy()
                if at_path_limit and not terminateds[0]:
                    effective_truncateds[0] = True
                    # next_buffer_observations is already correct:
                    #   naturally_done → already set to final_obs above
                    #   forced only   → next_observations IS the correct obs
                    #                   (env has not auto-reset)

                # Log raw env episode info if provided by the wrapper
                if "episode_info" in env_infos:
                    logger.update_metric(**env_infos["episode_info"])

                transition = {
                    "observation": observations,
                    "action": actions,
                    "reward": rewards,
                    "terminated": terminateds,
                    "truncated": effective_truncateds,
                    "next_observation": next_buffer_observations,
                }
                agent.process_transition(transition)  # also resamples skill at done

                episode_return += float(rewards[0])
                episode_length += 1
                prev_transition = {"next_observation": next_observations}
                observations = next_observations
                total_env_steps += 1

                if terminateds[0] or effective_truncateds[0]:
                    break

            episode_returns.append(episode_return)
            episode_lengths.append(episode_length)

        # ── Phase 2: trans_optimization_epochs gradient updates ───────────────
        # Buffer must have at least buffer_min_length samples first.
        if agent.can_start_training():
            for _update_idx in range(cfg.trans_optimization_epochs):
                update_info = agent.update()
                logger.update_metric(**update_info)
                logger.update_metric(**_remap_to_metra_keys(update_info))

        # Log per-epoch episode statistics
        logger.update_metric(**{
            "episode/mean_return": float(np.mean(episode_returns)),
            "episode/mean_length": float(np.mean(episode_lengths)),
            "train/total_env_steps": total_env_steps,
            "train/epoch": epoch + 1,
        })
        # Official METRA-compatible path stats
        logger.update_metric(**{
            "METRA/PathLengthMean": float(np.mean(episode_lengths)),
            "METRA/PathLengthMax":  float(np.max(episode_lengths)),
            "METRA/PathLengthMin":  float(np.min(episode_lengths)),
        })

        # ── Logging ───────────────────────────────────────────────────────────
        if (epoch + 1) % cfg.n_epochs_per_log == 0:
            logger.log_metric(step=total_env_steps)
            logger.reset()

        # ── Evaluation ────────────────────────────────────────────────────────
        if (epoch + 1) % cfg.n_epochs_per_eval == 0:
            eval_info = evaluate(agent, eval_env, cfg.num_eval_episodes, cfg.env.env_type)
            logger.update_metric(**eval_info)

        # ── Video recording ───────────────────────────────────────────────────
        if (epoch + 1) % cfg.n_epochs_per_record == 0:
            video_info = record_video(agent, record_env, cfg.num_record_episodes, cfg.env.env_type)
            logger.update_metric(**video_info)

        # ── Checkpoint ────────────────────────────────────────────────────────
        if (epoch + 1) % cfg.n_epochs_per_save == 0:
            save_path = os.path.join(save_path_base, f"epoch{epoch + 1}")
            agent.save(save_path)
            _prune_old_checkpoints(save_path_base, getattr(cfg, "max_checkpoints_to_keep", None))

    # ─────────────────────────────────────────────────────────────────────────
    # Final evaluation
    # ─────────────────────────────────────────────────────────────────────────
    eval_info = evaluate(agent, eval_env, cfg.num_eval_episodes, cfg.env.env_type)
    video_info = record_video(agent, record_env, cfg.num_record_episodes, cfg.env.env_type)
    logger.update_metric(**eval_info)
    logger.update_metric(**video_info)
    logger.log_metric(step=total_env_steps)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="METRA training aligned with official rhythm (epoch-based)."
    )
    parser.add_argument("--config-path", type=str, default="configs")
    parser.add_argument("--config-name", type=str, default="metra_aligned_base")
    parser.add_argument("overrides", nargs="*", default=[])
    return parser.parse_args()


if __name__ == "__main__":
    run(_parse_args())
