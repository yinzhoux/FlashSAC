from typing import Any, Optional

import torch
from torch.amp.grad_scaler import GradScaler
from .metra_config import METRAConfig
from flash_rl.agents.utils.network import Network
from flash_rl.buffers import Batch
from .metra_comp import (
    compute_metra_constraint,
    compute_metra_intrinsic_reward,
     compute_metra_reward
)

def add_prefix_to_keys(d: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {f"{prefix}/{k}": v for k, v in d.items()}

@torch.compile
def _select_min_q_log_probs(
    next_qs: torch.Tensor,  # (2, B)
    next_q_log_probs: torch.Tensor,  # (2, B, num_bins)
) -> torch.Tensor:
    """Select log-probs from the min-Q critic and return the next-obs half (batch, num_bins)."""
    num_bins = next_q_log_probs.shape[-1]
    min_indices = next_qs.argmin(dim=0)  # (B,)
    selected = torch.gather(
        next_q_log_probs,
        dim=0,
        index=min_indices[None, :, None].expand(1, -1, num_bins),
    )[
        0
    ]  # (B, num_bins)
    return selected


@torch.compile
def _compute_categorical_td_target(
    target_log_probs: torch.Tensor,  # (B, num_bins)
    reward: torch.Tensor,  # (B,)
    done: torch.Tensor,  # (B,)
    actor_entropy: torch.Tensor,  # (B,)
    gamma: float,
    num_bins: int,
    min_v: float,
    max_v: float,
) -> torch.Tensor:
    batch_size = reward.shape[0]

    reward = reward.reshape(-1, 1)
    done = done.reshape(-1, 1)
    actor_entropy = actor_entropy.reshape(-1, 1)

    # Compute target value buckets
    bin_width = (max_v - min_v) / (num_bins - 1)
    bin_values = torch.linspace(
        min_v, max_v, num_bins, device=target_log_probs.device, dtype=target_log_probs.dtype
    ).view(1, -1)

    # target_bin_values
    target_bin_values = reward + gamma * (bin_values - actor_entropy) * (1.0 - done)
    target_bin_values = torch.clamp(target_bin_values, min_v, max_v)

    # update indices
    b = (target_bin_values - min_v) / bin_width
    lower = torch.floor(b).long()
    upper = torch.clamp(lower + 1, 0, num_bins - 1)

    frac = b - lower.float()

    # Compute target probabilities using exp
    target_probs_exp = target_log_probs.exp()
    m_l = target_probs_exp * (1.0 - frac)
    m_u = target_probs_exp * frac

    # Allocate output tensor
    target_probs = torch.zeros(batch_size, num_bins, dtype=target_probs_exp.dtype, device=target_probs_exp.device)

    # Scatter operations
    target_probs.scatter_add_(1, lower, m_l)
    target_probs.scatter_add_(1, upper, m_u)

    return target_probs


def update_actor(
    actor: Network,
    critic: Network,
    temperature: Network,
    batch: Batch,
    bc_alpha: float,
    device: torch.device,
    use_amp: bool,
    grad_scaler: Optional[GradScaler],
) -> dict[str, torch.Tensor]:
    """Update actor network.

    Args:
        actor: Actor network.
        critic: Critic network.
        temperature: Temperature network.
        batch: Batch of transitions.
        bc_alpha: BC regularization coefficient.
        device: Device to use.
        use_amp: Whether to use automatic mixed precision.
        grad_scaler: GradScaler for FP16 AMP.
    """

    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        actor_obs_all = torch.cat([batch["actor_observation"], batch["actor_next_observation"]], dim=0)  # type: ignore
        actions_all, info = actor(
            observations=actor_obs_all,
            training=True,
        )
        log_probs_all = info["log_prob"]

        actions = torch.chunk(actions_all, 2, dim=0)[0]
        log_probs = torch.chunk(log_probs_all, 2, dim=0)[0]

        # Disable critic gradients to prevent CUDA graph overwriting
        critic.network.requires_grad_(False)
        qs, q_infos = critic(
            observations=batch["observation"],
            actions=actions,
            training=False,
        )
        q = torch.minimum(qs[0], qs[1])
        critic.network.requires_grad_(True)

        temp_value = temperature().detach()
        actor_loss = (log_probs * temp_value - q).mean()

        if bc_alpha > 0:
            # https://arxiv.org/abs/2306.02451
            q_abs = torch.abs(q).mean().detach()
            bc_loss = ((actions - batch["action"]) ** 2).mean()
            actor_loss = actor_loss + bc_alpha * q_abs * bc_loss

        entropy = -log_probs.mean()
        mean_action = actions.mean()

    # Gradient step
    assert actor.optimizer is not None
    actor.optimizer.zero_grad(set_to_none=True)
    if use_amp:
        assert grad_scaler is not None
        grad_scaler.scale(actor_loss).backward()
        grad_scaler.step(actor.optimizer)
        grad_scaler.update()
    else:
        actor_loss.backward()
        actor.optimizer.step()

    # LR scheduler
    if actor.scheduler is not None:
        actor.scheduler.step()

    # Weight Normalization
    # NOTE: Make sure you finish all computation before this (e.g., computing info values)
    actor.normalize_parameters()

    update_info = {
        "loss": actor_loss,
        "entropy": entropy,
        "mean_action": mean_action,
    }
    update_info = add_prefix_to_keys(update_info, "actor")

    return update_info


def update_critic(
    actor: Network,
    critic: Network,
    target_critic: Network,
    temperature: Network,
    batch: Batch,
    min_v: float,
    max_v: float,
    num_bins: int,
    gamma: float,
    n_step: int,
    device: torch.device,
    use_amp: bool,
    grad_scaler: Optional[GradScaler],
) -> dict[str, torch.Tensor]:
    """Update critic network.

    Args:
        actor: Actor network.
        critic: Critic network.
        target_critic: Target critic network.
        temperature: Temperature network.
        batch: Batch of transitions.
        min_v: Minimum value for categorical distribution.
        max_v: Maximum value for categorical distribution.
        num_bins: Number of bins for categorical distribution.
        gamma: Discount factor.
        n_step: N-step return.
        device: Device to use.
        use_amp: Whether to use automatic mixed precision.
        grad_scaler: GradScaler for FP16 AMP.
    """

    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        # Compute target values
        with torch.no_grad():
            next_actions, info = actor(
                observations=batch["actor_next_observation"],
                training=False,
            )
            # Clone variables to prevent overwriting
            next_actions = next_actions.clone()
            next_actor_log_probs = info["log_prob"].clone()

            temp_value = temperature()

            next_actor_entropy = temp_value * next_actor_log_probs
            obs_all = torch.cat([batch["observation"], batch["next_observation"]], dim=0)  # type: ignore
            act_all = torch.cat([batch["action"], next_actions], dim=0)  # type: ignore

            # qs_all: (2, 2*B)
            # q_infos_all['log_probs']: (2, 2*B, num_bins)
            qs_all, q_infos_all = target_critic(
                observations=obs_all,
                actions=act_all,
                training=True,
            )
            next_qs = qs_all.chunk(2, dim=1)[1]
            next_q_log_probs = q_infos_all["log_prob"].chunk(2, dim=1)[1]
            next_q_log_probs = _select_min_q_log_probs(next_qs, next_q_log_probs)

            # Compute target probs
            target_probs = _compute_categorical_td_target(
                target_log_probs=next_q_log_probs,
                reward=batch["reward"],  # type: ignore
                done=batch["terminated"],  # type: ignore
                actor_entropy=next_actor_entropy,
                gamma=gamma**n_step,
                num_bins=num_bins,
                min_v=min_v,
                max_v=max_v,
            )
            max_entropy_bonus = next_actor_entropy.max()

        # Compute predicted q-value
        pred_qs_all, pred_q_infos = critic(
            observations=obs_all,
            actions=act_all,
            training=True,
        )
        pred_log_probs = torch.chunk(pred_q_infos["log_prob"], 2, dim=1)[0]

        ce_loss = -(target_probs.unsqueeze(0) * pred_log_probs).sum(dim=-1)  # (2, B)
        critic_loss = ce_loss.mean()

    # Gradient step
    assert critic.optimizer is not None
    critic.optimizer.zero_grad(set_to_none=True)
    if use_amp:
        assert grad_scaler is not None
        grad_scaler.scale(critic_loss).backward()  # type: ignore
        grad_scaler.step(critic.optimizer)
        grad_scaler.update()
    else:
        critic_loss.backward()  # type: ignore
        critic.optimizer.step()

    # LR scheduler
    if critic.scheduler is not None:
        critic.scheduler.step()

    # Weight Normalization
    critic.normalize_parameters()

    update_info = {
        "loss": critic_loss,
        "max_entropy_bonus": max_entropy_bonus,
    }
    update_info = add_prefix_to_keys(update_info, "critic")

    return update_info


@torch.no_grad()
def update_target_network(
    target_network: Network,
) -> dict[str, torch.Tensor]:
    # Use prepared/compiled EMA function for update
    target_network.ema_update_parameters()
    info: dict[str, torch.Tensor] = {}
    return info


def update_temperature(
    temperature: Network,
    entropy: torch.Tensor,
    target_entropy: float,
) -> dict[str, torch.Tensor]:
    """Update temperature network.

    Args:
        temperature: Temperature network.
        entropy: Current entropy value.
        target_entropy: Target entropy value.
    """

    temperature_value = temperature().clone()
    temperature_loss = temperature_value * (entropy.detach() - target_entropy).mean()

    assert temperature.optimizer is not None
    temperature.optimizer.zero_grad(set_to_none=True)
    temperature_loss.backward()
    temperature.optimizer.step()
    if temperature.scheduler is not None:
        temperature.scheduler.step()

    update_info = {
        "value": temperature_value,
        "loss": temperature_loss,
    }
    update_info = add_prefix_to_keys(update_info, "temperature")

    return update_info

def update_skill_encoder(
    skill_encoder: Network,
    batch: dict[str, torch.Tensor],
    constraint_epsilon: float,
    lambda_value: float,
    device: torch.device,
    use_amp: bool,
    grad_scaler: Optional[GradScaler],
    parameter_normalization: bool = False
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    raw_observations      = batch["raw_observation"]
    raw_next_observations = batch["raw_next_observation"]
    skills                = batch["skill"]

    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        current_features = skill_encoder(observations=raw_observations, training=True)
        next_features    = skill_encoder(observations=raw_next_observations, training=True)

        # Clone compiled outputs before reuse to avoid overwritten CUDAGraph buffers.
        current_features = current_features.clone()
        next_features    = next_features.clone()

        intrinsic_reward, squared_distance = compute_metra_intrinsic_reward(
            current_features=current_features,
            next_features=next_features,
            skills=skills
        )
        constraint_term = compute_metra_constraint(
            squared_distance=squared_distance,
            epsilon=constraint_epsilon
        )
        skill_encoder_objective = intrinsic_reward + lambda_value * constraint_term
        skill_encoder_loss      = -skill_encoder_objective.mean()

    assert skill_encoder.optimizer is not None
    skill_encoder.optimizer.zero_grad(set_to_none=True)
    if use_amp:
        assert grad_scaler is not None
        grad_scaler.scale(skill_encoder_loss).backward()
        grad_scaler.step(skill_encoder.optimizer)
        grad_scaler.update()
    else:
        skill_encoder_loss.backward()
        skill_encoder.optimizer.step()

    if skill_encoder.scheduler is not None:
        skill_encoder.scheduler.step()

    if parameter_normalization:
        skill_encoder.normalize_parameters()

    update_info = {
        "mean_objective":        skill_encoder_objective.mean(),
        "mean_constraint":       constraint_term.mean(),
        "mean_intrinsic_reward": intrinsic_reward.mean(),
        "mean_squared_distance": squared_distance.mean(),
    }
    update_info = {f"skill_encoder/{key}": value for key, value in update_info.items()}
    return update_info, intrinsic_reward.detach(), constraint_term.detach()

def update_dual_lambda(
    dual_lambda: Network,
    constraint_term: torch.Tensor,
) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        constraint_term = constraint_term.detach()

    lambda_value = dual_lambda().clone()
    dual_lambda_loss = (lambda_value * constraint_term).mean()

    assert dual_lambda.optimizer is not None
    dual_lambda.optimizer.zero_grad(set_to_none=True)
    dual_lambda_loss.backward()
    dual_lambda.optimizer.step()

    if dual_lambda.scheduler is not None:
        dual_lambda.scheduler.step()

    updated_lambda = dual_lambda().detach().clone()
    return {
        "dual_lambda/loss": dual_lambda_loss.detach(),
        "dual_lambda/value": updated_lambda
    }

def update_policy(
    batch: dict[str, torch.Tensor],
    actor: Network,
    critic: Network,
    target_critic: Network,
    temperature: Network,
    cfg: METRAConfig,
    do_actor_update: bool,
    device: torch.device,
    grad_scaler: Optional[GradScaler],
):
    if do_actor_update:
        # Update actor
        actor_info = update_actor(
            actor=actor,
            critic=critic,
            temperature=temperature,
            batch=batch,  # type: ignore
            bc_alpha=cfg.actor_bc_alpha,
            device=device,
            use_amp=cfg.use_amp,
            grad_scaler=grad_scaler,
        )

        # Update temperature
        temperature_info = update_temperature(
            temperature=temperature,
            entropy=actor_info["actor/entropy"],
            target_entropy=cfg.temp_target_entropy,
        )
    else:
        actor_info = {}
        temperature_info = {}

    # Update critic
    critic_info = update_critic(
        actor=actor,  # updated
        critic=critic,
        target_critic=target_critic,
        temperature=temperature,  # updated
        batch=batch,  # type: ignore
        min_v=cfg.critic_min_v,
        max_v=cfg.critic_max_v,
        num_bins=cfg.critic_num_bins,
        gamma=cfg.gamma,
        n_step=cfg.n_step,
        device=device,
        use_amp=cfg.use_amp,
        grad_scaler=grad_scaler,
    )

    target_critic_info = update_target_network(
        target_network=target_critic,
    )

    # Merge all info dicts
    update_info = {
        **actor_info,
        **critic_info,
        **target_critic_info,
        **temperature_info,
    }

    return update_info
