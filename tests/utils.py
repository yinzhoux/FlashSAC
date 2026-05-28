import torch
import matplotlib.pyplot as plt


import os
import sys

sys.path.append(os.path.abspath('..'))

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
os.environ["MUJOCO_GL"] = 'egl'
import argparse
import random
import shutil
from datetime import datetime
from typing import Optional

import hydra
import numpy as np
import torch
import tqdm
from omegaconf import DictConfig, OmegaConf, open_dict

from flash_rl.agents import create_agent
from flash_rl.common import create_logger
from flash_rl.envs import create_envs
from flash_rl.evaluation import evaluate, record_video
from flash_rl.types import Tensor
from plot_metra_eval import generate_eval_artifacts


normalizer_mean = np.array(
    [0.00486117, 0.011312, 0.7022248, 0.8454677, -0.00102548, -0.00300276, 0.00311523, -0.00139029,
        0.8607109, -0.00185301, -0.8556998, 0.00343217, -0.8585605, -0.00109082, 0.8558013, 0.00278213,
        0.00618173, -0.02584622, -0.00599026, -0.00379596, 0.00526138, -0.0059213, 0.27686235, 0.00512205,
        -0.27617684, -0.0033233, -0.2766923, 0.00268359, 0.27756855])
normalizer_std = np.array(
    [0.62473416, 0.61958003, 0.1717569, 0.28629342, 0.20020866, 0.20572574, 0.34922406, 0.40098143,
        0.3114514, 0.4024826, 0.31057045, 0.40343934, 0.3110796, 0.40245822, 0.31100526, 0.81786263, 0.8166509,
        0.9870919, 1.7525449, 1.7468817, 1.8596431, 4.502961, 4.4070187, 4.522444, 4.3518476, 4.5105968,
        4.3704205, 4.5175962, 4.3704395])

import matplotlib.pyplot as plt

def plot_fix_batch_history(history, show=True, path=None):
    steps = [x["step"] for x in history]

    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    axes = axes.flatten()

    axes[0].plot(steps, [x["reward_mean"] for x in history], label="reward_mean")
    axes[0].plot(steps, [x["reward_abs_mean"] for x in history], label="reward_abs_mean")
    axes[0].set_title("Reward")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(steps, [x["squared_dist_mean"] for x in history], label="squared_dist_mean")
    axes[1].plot(steps, [x["squared_dist_gt_1_ratio"] for x in history], label="squared_dist > 1 ratio")
    axes[1].set_title("Squared Distance")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(steps, [x["constraint_mean"] for x in history], label="constraint_mean")
    axes[2].plot(steps, [x["constraint_neg_ratio"] for x in history], label="constraint < 0 ratio")
    axes[2].plot(steps, [x["constraint_eq_eps_ratio"] for x in history], label="constraint == eps ratio")
    axes[2].set_title("Constraint")
    axes[2].legend()
    axes[2].grid(True)

    axes[3].plot(steps, [x["lambda_before"] for x in history], label="lambda_before")
    axes[3].plot(steps, [x["lambda_after"] for x in history], label="lambda_after")
    axes[3].set_title("Dual Lambda")
    axes[3].legend()
    axes[3].grid(True)

    axes[4].plot(steps, [x["te_obj_mean"] for x in history], label="te_obj_mean")
    axes[4].plot(steps, [x["skill_encoder_loss"] for x in history], label="skill_encoder_loss")
    axes[4].plot(steps, [x["dual_lambda_loss"] for x in history], label="dual_lambda_loss")
    axes[4].set_title("Objective / Loss")
    axes[4].legend()
    axes[4].grid(True)

    axes[5].plot(steps, [x["reward_abs_mean"] for x in history], label="reward_abs_mean")
    axes[5].plot(steps, [x["squared_dist_mean"] for x in history], label="squared_dist_mean")
    axes[5].set_title("Scale Comparison")
    axes[5].legend()
    axes[5].grid(True)

    axes[6].plot(steps, [x["constraint_neg_ratio"] for x in history], label="constraint < 0 ratio")
    axes[6].plot(steps, [x["squared_dist_gt_1_ratio"] for x in history], label="squared_dist > 1 ratio")
    axes[6].set_title("Tail Behavior")
    axes[6].legend()
    axes[6].grid(True)

    axes[7].plot(steps, [x["reward_mean"] for x in history], label="reward_mean")
    axes[7].plot(steps, [x["lambda_constraint_mean"] for x in history], label="lambda * constraint mean")
    axes[7].plot(steps, [x["te_obj_mean"] for x in history], label="te_obj_mean")
    axes[7].set_title("Who Dominates Objective")
    axes[7].legend()
    axes[7].grid(True)

    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig(path)

def build_history_from_logs(logs, eps=1e-3):
    history = {
        "reward_mean": [],
        "reward_abs_mean": [],
        "reward_std": [],
        "squared_dist_mean": [],
        "squared_dist_gt_1_ratio": [],
        "constraint_mean": [],
        "constraint_neg_ratio": [],
        "constraint_eq_eps_ratio": [],
        "lambda": [],
        "lambda_constraint_mean": [],
        "te_obj_mean": [],
        "skill_encoder_loss": [],
    }

    for log in logs:
        intrinsic_reward, squared_distance, constraint_term, lambda_value, skill_encoder_loss = log

        intrinsic_reward = intrinsic_reward.detach().float().cpu()
        squared_distance = squared_distance.detach().float().cpu()
        constraint_term = constraint_term.detach().float().cpu()

        if torch.is_tensor(lambda_value):
            lambda_scalar = float(lambda_value.detach().float().cpu().item())
        else:
            lambda_scalar = float(lambda_value)

        if torch.is_tensor(skill_encoder_loss):
            loss_scalar = float(skill_encoder_loss.detach().float().cpu().item())
        else:
            loss_scalar = float(skill_encoder_loss)

        te_obj = intrinsic_reward + lambda_scalar * constraint_term

        history["reward_mean"].append(intrinsic_reward.mean().item())
        history["reward_abs_mean"].append(intrinsic_reward.abs().mean().item())
        history["reward_std"].append(intrinsic_reward.std().item())

        history["squared_dist_mean"].append(squared_distance.mean().item())
        history["squared_dist_gt_1_ratio"].append((squared_distance > 1.0).float().mean().item())

        history["constraint_mean"].append(constraint_term.mean().item())
        history["constraint_neg_ratio"].append((constraint_term < 0.0).float().mean().item())
        history["constraint_eq_eps_ratio"].append(torch.isclose(
            constraint_term,
            torch.full_like(constraint_term, eps),
            atol=1e-6,
        ).float().mean().item())

        history["lambda"].append(lambda_scalar)
        history["lambda_constraint_mean"].append((lambda_scalar * constraint_term).mean().item())
        history["te_obj_mean"].append(te_obj.mean().item())
        history["skill_encoder_loss"].append(loss_scalar)

    return history
def plot_history(history):
    steps = list(range(len(history["reward_mean"])))

    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    axes = axes.flatten()

    axes[0].plot(steps, history["reward_mean"], label="reward_mean")
    axes[0].plot(steps, history["reward_abs_mean"], label="reward_abs_mean")
    axes[0].fill_between(
        steps,
        [m - s for m, s in zip(history["reward_mean"], history["reward_std"])],
        [m + s for m, s in zip(history["reward_mean"], history["reward_std"])],
        alpha=0.2,
        label="reward_mean ± std",
    )
    axes[0].set_title("Reward")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(steps, history["squared_dist_mean"], label="squared_dist_mean")
    axes[1].plot(steps, history["squared_dist_gt_1_ratio"], label="squared_dist > 1 ratio")
    axes[1].set_title("Squared Distance")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(steps, history["constraint_mean"], label="constraint_mean")
    axes[2].plot(steps, history["constraint_neg_ratio"], label="constraint < 0 ratio")
    axes[2].plot(steps, history["constraint_eq_eps_ratio"], label="constraint == eps ratio")
    axes[2].set_title("Constraint")
    axes[2].legend()
    axes[2].grid(True)

    axes[3].plot(steps, history["lambda"], label="lambda")
    axes[3].plot(steps, history["lambda_constraint_mean"], label="lambda * constraint mean")
    axes[3].set_title("Dual")
    axes[3].legend()
    axes[3].grid(True)

    axes[4].plot(steps, history["te_obj_mean"], label="te_obj_mean")
    axes[4].plot(steps, history["skill_encoder_loss"], label="skill_encoder_loss")
    axes[4].set_title("Objective / Loss")
    axes[4].legend()
    axes[4].grid(True)

    axes[5].plot(steps, history["reward_abs_mean"], label="reward_abs_mean")
    axes[5].plot(steps, history["squared_dist_mean"], label="squared_dist_mean")
    axes[5].set_title("Scale Comparison")
    axes[5].legend()
    axes[5].grid(True)

    axes[6].plot(steps, history["constraint_neg_ratio"], label="constraint < 0 ratio")
    axes[6].plot(steps, history["squared_dist_gt_1_ratio"], label="squared_dist > 1 ratio")
    axes[6].set_title("Tail Behavior")
    axes[6].legend()
    axes[6].grid(True)

    axes[7].plot(steps, history["reward_mean"], label="reward_mean")
    axes[7].plot(steps, history["lambda_constraint_mean"], label="lambda * constraint mean")
    axes[7].plot(steps, history["te_obj_mean"], label="te_obj_mean")
    axes[7].set_title("Who Dominates Objective")
    axes[7].legend()
    axes[7].grid(True)

    plt.tight_layout()
    plt.show()


def _save_eval_artifacts(
    agent: object,
    record_env: object,
    cfg: DictConfig,
    save_path_base: str,
    *,
    label: str,
) -> None:
    artifact_dir = os.path.join(save_path_base, "eval_artifacts", label)
    artifact_kwargs = dict(
        num_random_trajectories=int(getattr(cfg, "num_random_trajectories", 48)),
        num_video_repeats=int(getattr(cfg, "num_video_repeats", 2)),
        video_fps=int(getattr(cfg, "video_fps", 15)),
        video_skip_frames=int(getattr(cfg, "video_skip_frames", 1)),
        eval_plot_axis=(list(getattr(cfg, "eval_plot_axis")) if getattr(cfg, "eval_plot_axis", None) is not None else None),
    )
    paths = generate_eval_artifacts(
        agent=agent,
        env=record_env,
        cfg=cfg,
        output_dir=artifact_dir,
        **artifact_kwargs,
    )
    print(
        "Saved eval artifacts:",
        f"phi={paths['phi_plot_path']}",
        f"traj={paths['traj_plot_path']}",
        f"video={paths['video_path']}",
    )

def _apply_normalize_obs(obs, mean, var):
    normalized_obs = (obs - mean) / (np.sqrt(var) + 1e-8)
    return normalized_obs.astype(np.float32)


# one epoch training test
def sample_one_path(train_env, agent, cfg, obs_norm=False):
    observations, _  = train_env.reset()
    prev_transition = {"next_observation": observations}
    episode_return = 0.0
    episode_length = 0

    for step in range(cfg.max_path_length):
        actions = agent.sample_actions(step, prev_transition, training=True) # training flag controls the randomness of sampling.
        actions = np.array(actions)

        actions = np.squeeze(actions, axis=0)

        next_observations, rewards, terminateds, truncateds, env_infos = train_env.step(actions)
        next_buffer_observations = next_observations.copy()
        naturally_done = terminateds or truncateds
        if naturally_done:
            next_buffer_observations[0] = env_infos["final_obs"][0]
        
        at_path_limit = (step == cfg.max_path_length - 1)
        effective_truncateds = truncateds
        if at_path_limit and not terminateds:
            effective_truncateds = True
        
        transition = {
            "observation": observations,
            'action': actions,
            "reward": rewards,
            "terminated": terminateds,
            "truncated": effective_truncateds,
            "next_observation": next_buffer_observations
        }

        agent.process_transition(transition)
        episode_return += float(rewards)
        episode_length += 1
        prev_transition = {"next_observation": next_observations}
        observations = next_observations
        
        if terminateds or effective_truncateds:
            break

    return episode_length, episode_return

from flash_rl.buffers.metra_buffer import METRATorchBuffer

def sample_from_buffer(buffer: METRATorchBuffer, device, return_full_batch: bool = False):
    batch = buffer.sample()
    batch = {key: value.to(device) for key, value in batch.items()}

    if return_full_batch:
        return batch

    return batch["observation"], batch["next_observation"], batch["skill"]

from flash_rl.agents.metra.metra_comp import compute_metra_constraint, compute_metra_intrinsic_reward, compute_metra_reward
from flash_rl.agents.metra.update import update_dual_lambda, update_policy, update_skill_encoder
from flash_rl.agents.utils.function import concat_obs_skill

def evaluate(observations, next_observations, skills, agent, training=False, printi=False):
    encoder_observations = torch.cat([observations, next_observations], dim=0)
    encoded_features = agent._skill_encoder(encoder_observations, training=training)
    current_features, next_features = torch.chunk(encoded_features, 2, dim=0)

    intrinsic_reward, squared_distance = compute_metra_intrinsic_reward(current_features=current_features, next_features=next_features, skills=skills)
    constraint_term = compute_metra_constraint(squared_distance=squared_distance, epsilon=1e-3)
    lambda_value = agent._dual_lambda().detach()
    skill_encoder_objective = intrinsic_reward + lambda_value * constraint_term
    skill_encoder_loss = -skill_encoder_objective.mean()    
    if printi:
        print("reward mean:", intrinsic_reward.mean().item())
        print("reward abs mean:", intrinsic_reward.abs().mean().item())
        print("reward std:", intrinsic_reward.std().item())
        print("constraint mean:", constraint_term.mean().item())
        print("constraint < 0 ratio:", (constraint_term < 0).float().mean().item())
        print("constraint == eps ratio:", (constraint_term == 1e-3).float().mean().item())
        print("squared_dist mean:", squared_distance.mean().item())
        print("squared_dist > 1 ratio:", (squared_distance > 1).float().mean().item())
        print("te obj mean:", skill_encoder_objective.mean().item())
        print("skill_encoder_loss:", skill_encoder_loss.item())

    return intrinsic_reward, squared_distance, constraint_term, lambda_value, skill_encoder_loss

def fix_batch_train_encoder(agent, obs, next_obs, skills, update_times: int = 50):
    logs = []
    for i in tqdm.tqdm(range(update_times)):
        log = evaluate(obs, next_obs, skills, agent, training=True, printi=False)
        logs.append(log)

        intrinsic_reward, squared_distance, constraint_term, lambda_value, skill_encoder_loss = log

        agent._skill_encoder.optimizer.zero_grad(set_to_none=True)
        skill_encoder_loss.backward()
        agent._skill_encoder.optimizer.step()

        if agent._skill_encoder.scheduler is not None:
            agent._skill_encoder.scheduler.step()


    return logs

def concat_obs_skill(observations: torch.Tensor, skills: torch.Tensor) -> torch.Tensor:
    return torch.cat([observations, skills], dim=-1)


def fix_batch_train_agent(agent, batch, update_times: int = 50, eps: float = 1e-3):
    history = []

    obs = batch["observation"]
    next_obs = batch["next_observation"]
    skills = batch["skill"]

    for i in range(update_times):
        log = evaluate(obs, next_obs, skills, agent, training=True, printi=False)

        intrinsic_reward, squared_distance, constraint_term, lambda_value, skill_encoder_loss = log

        reward_mean = intrinsic_reward.detach().mean().item()
        reward_abs_mean = intrinsic_reward.detach().abs().mean().item()
        reward_std = intrinsic_reward.detach().std().item()

        squared_dist_mean = squared_distance.detach().mean().item()
        squared_dist_gt_1_ratio = (squared_distance.detach() > 1.0).float().mean().item()

        constraint_mean = constraint_term.detach().mean().item()
        constraint_neg_ratio = (constraint_term.detach() < 0.0).float().mean().item()
        constraint_eq_eps_ratio = torch.isclose(
            constraint_term.detach(),
            torch.full_like(constraint_term.detach(), eps),
            atol=1e-6,
        ).float().mean().item()

        if torch.is_tensor(lambda_value):
            lambda_scalar_before = lambda_value.detach().item()
        else:
            lambda_scalar_before = float(lambda_value)

        lambda_constraint_mean = (lambda_scalar_before * constraint_term.detach()).mean().item()
        te_obj_mean = (intrinsic_reward.detach() + lambda_scalar_before * constraint_term.detach()).mean().item()
        skill_encoder_loss_scalar = skill_encoder_loss.detach().item()

        agent._skill_encoder.optimizer.zero_grad(set_to_none=True)
        skill_encoder_loss.backward()
        agent._skill_encoder.optimizer.step()

        if agent._skill_encoder.scheduler is not None:
            agent._skill_encoder.scheduler.step()

        with torch.no_grad():
            cst_mean = constraint_term.detach().mean()

        log_lambda = agent._dual_lambda.network.log_temp
        dual_lambda_loss = log_lambda * cst_mean

        assert agent._dual_lambda.optimizer is not None
        agent._dual_lambda.optimizer.zero_grad(set_to_none=True)
        dual_lambda_loss.backward()
        agent._dual_lambda.optimizer.step()

        if agent._dual_lambda.scheduler is not None:
            agent._dual_lambda.scheduler.step()

        lambda_scalar_after = agent._dual_lambda().detach().item()
        dual_lambda_loss_scalar = dual_lambda_loss.detach().item()

        # Update policy
        with torch.no_grad():
            log = evaluate(obs, next_obs, skills, agent, training=True, printi=False)
            intrinsic_reward, squared_distance, constraint_term, lambda_value, skill_encoder_loss = log
            batch["observation"] = concat_obs_skill(obs, skills)
            batch["next_observation"] = concat_obs_skill(next_obs, skills)
            batch["actor_observation"] = concat_obs_skill(obs, skills)
            batch["actor_next_observation"] = concat_obs_skill(
                next_obs,
                skills,
            )

            batch["reward"] = intrinsic_reward

        _update_info = update_policy(
            batch=batch,
            actor=agent._actor,
            critic=agent._critic,
            target_critic=agent._target_critic,
            temperature=agent._temperature,
            cfg=agent._cfg,
            do_actor_update=(agent._update_step % agent._cfg.actor_update_period == 0),
            device=agent._device,
            grad_scaler=agent._grad_scaler,
        )

        agent._update_step += 1

        history.append({
            "step": i,
            "reward_mean": reward_mean,
            "reward_abs_mean": reward_abs_mean,
            "reward_std": reward_std,
            "squared_dist_mean": squared_dist_mean,
            "squared_dist_gt_1_ratio": squared_dist_gt_1_ratio,
            "constraint_mean": constraint_mean,
            "constraint_neg_ratio": constraint_neg_ratio,
            "constraint_eq_eps_ratio": constraint_eq_eps_ratio,
            "lambda_before": lambda_scalar_before,
            "lambda_after": lambda_scalar_after,
            "lambda_constraint_mean": lambda_constraint_mean,
            "te_obj_mean": te_obj_mean,
            "skill_encoder_loss": skill_encoder_loss_scalar,
            "dual_lambda_loss": dual_lambda_loss_scalar,
        })

    return history