import os
import argparse
import shutil
from datetime import datetime
import hydra
from omegaconf import OmegaConf, open_dict
import random
import torch
import numpy as np
from flash_rl.agents.metra.sampler import Sampler
from flash_rl.agents.metra.agent import METRAAgent
import tqdm
from flash_rl.agents.metra.update import train_once
from flash_rl.common import create_logger

def get_argparser():
    '''
    Construct args parser.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./configs')
    parser.add_argument('--config_name', type=str, default='general_base')
    parser.add_argument('--overrides', action='append', default=[])
    return parser

def resolve_cfg(config_path: str, config_name: str, overrides: list):
    OmegaConf.register_new_resolver("eval", lambda s: eval(s))
    hydra.initialize(version_base=None, config_path=config_path)
    cfg = hydra.compose(config_name=config_name, overrides=overrides)
    OmegaConf.resolve(cfg)
    return cfg


def ensure_logging_cfg(cfg):
    with open_dict(cfg):
        if "project_name" not in cfg:
            cfg.project_name = "Metra"
        if "entity_name" not in cfg:
            cfg.entity_name = "local"
        if "group_name" not in cfg:
            cfg.group_name = "general"
        if "exp_name" not in cfg:
            cfg.exp_name = cfg.config_name if "config_name" in cfg else "general_train"
        if "logger_type" not in cfg:
            cfg.logger_type = "tensorboard"
        if "save_path" not in cfg:
            cfg.save_path = "models/${group_name}/${exp_name}/seed${seed}-TIMESTAMP"
        if "env" not in cfg:
            cfg.env = OmegaConf.create({"env_name": "Ant-v4"})
        elif "env_name" not in cfg.env:
            cfg.env.env_name = "Ant-v4"
        if "logging_per_epoch" not in cfg:
            cfg.logging_per_epoch = 1
        if "save_checkpoint_per_epoch" not in cfg:
            cfg.save_checkpoint_per_epoch = None
        if "save_final_checkpoint" not in cfg:
            cfg.save_final_checkpoint = True
        if "max_checkpoints_to_keep" not in cfg:
            cfg.max_checkpoints_to_keep = None


def to_scalar(value):
    if isinstance(value, (float, int, np.floating, np.integer)):
        return float(value)
    if isinstance(value, torch.Tensor):
        detached = value.detach()
        if detached.numel() == 1:
            return float(detached.cpu().item())
        return float(detached.float().mean().cpu().item())
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return float(value.item())
        return float(value.mean())
    return None


def collect_training_metrics(agent, training_info, data, epoch_idx: int, update_idx: int):
    metrics = {
        "train/epoch": epoch_idx,
        "train/update": update_idx,
        "buffer/size": getattr(agent.replay_buffer, "_num_in_buffer", 0),
        "train/reward_mean": data["reward"].mean(),
        "train/reward_abs_mean": data["reward"].abs().mean(),
        "train/dual_lambda": training_info.get("dual_lam"),
        "train/loss_te": training_info.get("loss_te"),
        "train/loss_dual_lam": training_info.get("loss_dual_lam"),
        "train/square_dist_mean": training_info.get("square_dist"),
        "train/cst_penalty_mean": training_info.get("cst_penalty"),
        "train/q_target_mean": training_info.get("q_target_mean"),
        "train/q_err_mean": training_info.get("q_err_mean"),
        "train/loss_qf1": training_info.get("loss_qf1"),
        "train/loss_qf2": training_info.get("loss_qf2"),
        "train/loss_policy": training_info.get("loss_policy"),
        "train/loss_alpha": training_info.get("loss_alpha"),
        "train/log_prob_mean": training_info.get("new_action_log_probs"),
        "train/alpha": agent.log_alpha.param.exp(),
    }
    scalar_metrics = {}
    for key, value in metrics.items():
        scalar = to_scalar(value)
        if scalar is not None:
            scalar_metrics[key] = scalar
    return scalar_metrics


def resolve_save_dir(cfg):
    timestamp = datetime.now().strftime("%m%d-%H%M%S")
    save_path = str(cfg.save_path).replace("TIMESTAMP", timestamp)
    if "${" in save_path:
        env_name = cfg.env.env_name if "env" in cfg and "env_name" in cfg.env else "Ant-v4"
        save_path = f"models/{cfg.group_name}/{cfg.exp_name}/{env_name}/seed{cfg.seed}-{timestamp}"
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), save_path)


def write_resolved_config(save_dir: str, cfg) -> None:
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "resolved_config.yaml"), "w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg))


def _count_parameters(module: torch.nn.Module) -> tuple[int, int]:
    total_params = sum(param.numel() for param in module.parameters())
    trainable_params = sum(param.numel() for param in module.parameters() if param.requires_grad)
    return total_params, trainable_params


def write_model_architecture(save_dir: str, agent: METRAAgent) -> None:
    modules = {
        "option_policy": agent.option_policy,
        "qf1": agent.qf1,
        "qf2": agent.qf2,
        "target_qf1": agent.target_qf1,
        "target_qf2": agent.target_qf2,
        "skill_encoder": agent.skill_encoder,
        "dual_lam": agent.dual_lam,
        "log_alpha": agent.log_alpha,
    }

    lines: list[str] = []
    for name, module in modules.items():
        total_params, trainable_params = _count_parameters(module)
        lines.append(f"[{name}]")
        lines.append(f"total_params: {total_params}")
        lines.append(f"trainable_params: {trainable_params}")
        lines.append(str(module))
        lines.append("")

    with open(os.path.join(save_dir, "model_architecture.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def save_checkpoint(agent: METRAAgent, save_dir: str, epoch: int, global_update_step: int) -> str:
    checkpoint_dir = os.path.join(save_dir, f"epoch{epoch}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "agent_state.pt")
    checkpoint = {
        "epoch": epoch,
        "global_update_step": global_update_step,
        "option_policy_state_dict": agent.option_policy.state_dict(),
        "qf1_state_dict": agent.qf1.state_dict(),
        "qf2_state_dict": agent.qf2.state_dict(),
        "target_qf1_state_dict": agent.target_qf1.state_dict(),
        "target_qf2_state_dict": agent.target_qf2.state_dict(),
        "skill_encoder_state_dict": agent.skill_encoder.state_dict(),
        "dual_lam_state_dict": agent.dual_lam.state_dict(),
        "log_alpha_state_dict": agent.log_alpha.state_dict(),
        "optimizer_state_dicts": {
            key: optimizer.state_dict() for key, optimizer in agent.optimizers.items()
        },
        "replay_buffer_size": getattr(agent.replay_buffer, "_num_in_buffer", 0),
    }
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_dir


def prune_old_checkpoints(save_dir: str, max_checkpoints_to_keep) -> None:
    if max_checkpoints_to_keep is None or max_checkpoints_to_keep <= 0:
        return

    checkpoint_dirs: list[tuple[int, str]] = []
    if not os.path.isdir(save_dir):
        return

    for name in os.listdir(save_dir):
        if not name.startswith("epoch"):
            continue
        epoch_str = name[5:]
        if not epoch_str.isdigit():
            continue
        checkpoint_dirs.append((int(epoch_str), os.path.join(save_dir, name)))

    checkpoint_dirs.sort(key=lambda item: item[0], reverse=True)
    for _, path in checkpoint_dirs[max_checkpoints_to_keep:]:
        shutil.rmtree(path, ignore_errors=True)

def set_random(seed):
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")  

def make_envs(cfg):
    '''
    @return:
        training env
    '''
    from flash_rl.envs.envs.mujoco.ant_env import AntEnv
    return AntEnv(render_hw=100, expose_all_qpos=cfg.env_pos_info, obs_norm=cfg.obs_norm)

# ==================CONFIG====================
# environment variable for mujoco record.
os.environ["MUJOCO_GL"] = 'egl'
# ============================================

def run(config_path, config_name, overrides):
    cfg = resolve_cfg(config_path, config_name, overrides)
    with open_dict(cfg):
        cfg.config_name = config_name
    ensure_logging_cfg(cfg)
    set_random(cfg.seed)
    train_env = make_envs(cfg)
    logger = create_logger(cfg)
    save_dir = resolve_save_dir(cfg)
    write_resolved_config(save_dir, cfg)

    agent = METRAAgent(train_env, cfg.agent)
    write_model_architecture(save_dir, agent)
    sampler = Sampler(train_env, cfg.agent.skill_dim)

    while not agent.replay_buffer.can_sample():
        path = sampler.rollout(agent.option_policy, path_len=200, training=True)
        agent.replay_buffer.add(path)

    logger.update_metric(**{"buffer/initial_size": float(agent.replay_buffer._num_in_buffer)})
    logger.log_metric(step=0)
    logger.reset()

    global_update_step = 0
    for epoch_idx in tqdm.tqdm(range(cfg.epochs), desc="epochs"):
        for update_idx in range(cfg.updates_per_epoch):
            training_info, data = train_once(agent)
            scalar_metrics = collect_training_metrics(agent, training_info, data, epoch_idx, global_update_step)
            logger.update_metric(**scalar_metrics)
            global_update_step += 1

        for _ in range(cfg.paths_per_epoch):
            path = sampler.rollout(agent.option_policy, path_len=200, training=True)
            agent.replay_buffer.add(path)

        if cfg.logging_per_epoch and ((epoch_idx + 1) % cfg.logging_per_epoch == 0):
            logger.update_metric(**{"buffer/size": float(agent.replay_buffer._num_in_buffer)})
            logger.log_metric(step=global_update_step)
            logger.reset()

        if cfg.save_checkpoint_per_epoch and ((epoch_idx + 1) % cfg.save_checkpoint_per_epoch == 0):
            save_checkpoint(agent, save_dir, epoch_idx + 1, global_update_step)
            prune_old_checkpoints(save_dir, cfg.max_checkpoints_to_keep)

    logger.update_metric(**{"buffer/final_size": float(agent.replay_buffer._num_in_buffer)})
    logger.log_metric(step=global_update_step)
    logger.reset()

    if cfg.save_final_checkpoint:
        save_checkpoint(agent, save_dir, cfg.epochs, global_update_step)
        prune_old_checkpoints(save_dir, cfg.max_checkpoints_to_keep)

if __name__ == "__main__":
    parser = get_argparser()
    args = parser.parse_args()
    run(args.config_path, args.config_name, args.overrides)
