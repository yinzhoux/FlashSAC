import torch
from .agent import METRAConfig
from .network import SkillEncoder
from flash_rl.agents.utils.network import Network
import torch.optim as optim
from flash_rl.agents.metra.network import (
    FlashSACActor,
    FlashSACDoubleCritic,
    FlashSACTemperature
)

def compute_metra_intrinsic_reward(
    current_features: torch.Tensor,
    next_features: torch.Tensor,
    skills: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    delta_features = next_features - current_features
    alignment = torch.sum(delta_features * skills, dim=-1)
    squared_distance = (delta_features**2).mean(dim=-1)
    return alignment, squared_distance


def compute_metra_constraint(
    squared_distance: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    return torch.minimum(
        torch.full_like(squared_distance, epsilon),
        1.0 - squared_distance,
    )

def compute_metra_reward(
        current_features: torch.Tensor,
        next_features: torch.Tensor,
        skills: torch.Tensor,
        epsilon: float,
        lambda_value: float
):
    alignment, squared_distance = compute_metra_intrinsic_reward(
        current_features, next_features, skills
    )

    constraint_term = compute_metra_constraint(
        squared_distance, epsilon
    )

    return alignment + constraint_term * lambda_value


def init_metra_networks(
    actor_observation_dim: int,
    critic_observation_dim: int,
    action_dim: int,
    skill_dim: int,
    skill_encoder_hidden_dim: int,
    skill_encoder_num_layers: int,
    cfg: METRAConfig,
    device: torch.device,
) -> tuple[Network, Network, Network, Network, Network, Network]:
    # Initialize actor
    actor_net = FlashSACActor(
        num_blocks=cfg.actor_num_blocks,
        input_dim=actor_observation_dim,
        hidden_dim=cfg.actor_hidden_dim,
        action_dim=action_dim,
    ).to(device)

    use_fused = device.type == "cuda" and torch.cuda.is_available()
    actor_optimizer = optim.Adam(actor_net.parameters(), lr=cfg.actor_learning_rate, fused=use_fused)
    actor_scheduler = torch.optim.lr_scheduler.ConstantLR(
        optimizer=actor_optimizer, factor=1.0
    )
    actor = Network(
        network=actor_net,
        optimizer=actor_optimizer,
        scheduler=actor_scheduler,
        compile_network=cfg.use_compile,
        compile_mode=cfg.compile_mode,
        use_weight_normalization=True,
    )
    # Manually compile `get_mean_and_std` function
    if cfg.use_compile:
        actor.network.get_mean_and_std = torch.compile(actor.network.get_mean_and_std, mode=cfg.compile_mode)  # type: ignore

    # Initialize critic
    critic_net = FlashSACDoubleCritic(
        num_blocks=cfg.critic_num_blocks,
        input_dim=critic_observation_dim + action_dim,
        hidden_dim=cfg.critic_hidden_dim,
        num_bins=cfg.critic_num_bins,
        min_v=cfg.critic_min_v,
        max_v=cfg.critic_max_v,
    ).to(device)

    critic_optimizer = optim.Adam(
        critic_net.parameters(),
        lr=cfg.critic_learning_rate,
        fused=use_fused,
    )
    critic_scheduler = torch.optim.lr_scheduler.ConstantLR(
        optimizer=critic_optimizer, factor=1.0
    )
    critic = Network(
        network=critic_net,
        optimizer=critic_optimizer,
        scheduler=critic_scheduler,
        compile_network=cfg.use_compile,
        compile_mode=cfg.compile_mode,
        use_weight_normalization=True,
    )

    # Initialize target critic (same as critic but no optimizer)
    target_critic_net = FlashSACDoubleCritic(
        num_blocks=cfg.critic_num_blocks,
        input_dim=critic_observation_dim + action_dim,
        hidden_dim=cfg.critic_hidden_dim,
        num_bins=cfg.critic_num_bins,
        min_v=cfg.critic_min_v,
        max_v=cfg.critic_max_v,
    ).to(device)
    target_critic_net.load_state_dict(critic_net.state_dict())
    target_critic = Network(
        network=target_critic_net,
        optimizer=None,
        scheduler=None,
        compile_network=cfg.use_compile,
        compile_mode=cfg.compile_mode,
        use_weight_normalization=True,
        ema_source=critic,  # wire EMA update source
        ema_tau=cfg.critic_target_update_tau,
    )

    # Initialize temperature
    temp_net = FlashSACTemperature(cfg.temp_initial_value).to(device)
    temp_optimizer = optim.Adam(
        temp_net.parameters(),
        lr=cfg.temp_learning_rate,
        fused=use_fused,
    )
    temp_scheduler = torch.optim.lr_scheduler.ConstantLR(
        optimizer=temp_optimizer, factor=1.0
    )
    temperature = Network(
        network=temp_net,
        optimizer=temp_optimizer,
        scheduler=temp_scheduler,
        compile_network=cfg.use_compile,
        compile_mode=cfg.compile_mode,
        use_weight_normalization=False,
    )

    skill_encoder_net = SkillEncoder(
        obs_dim=actor_observation_dim - skill_dim,
        skill_dim=skill_dim,
        hidden_dim=skill_encoder_hidden_dim,
        num_layers=skill_encoder_num_layers,
    ).to(device)
    skill_encoder_optimizer = optim.Adam(
        skill_encoder_net.parameters(),
        lr=cfg.skill_encoder_learning_rate,
        fused=use_fused,
    )
    skill_encoder_scheduler = torch.optim.lr_scheduler.ConstantLR(
        optimizer=skill_encoder_optimizer, factor=1.0
    )
    skill_encoder = Network(
        network=skill_encoder_net,
        optimizer=skill_encoder_optimizer,
        scheduler=skill_encoder_scheduler,
        compile_network=cfg.use_compile,
        compile_mode=cfg.compile_mode,
        use_weight_normalization=True,
    )

    dual_lambda_net = FlashSACTemperature(cfg.dual_lambda_init_value).to(device)
    dual_lambda_optimizer = optim.Adam(
        dual_lambda_net.parameters(),
        lr=cfg.dual_lambda_learning_rate,
        fused=use_fused,
    )
    dual_lambda_scheduler = torch.optim.lr_scheduler.ConstantLR(
        optimizer=dual_lambda_optimizer, factor=1.0
    )
    dual_lambda = Network(
        network=dual_lambda_net,
        optimizer=dual_lambda_optimizer,
        scheduler=dual_lambda_scheduler,
        compile_network=cfg.use_compile,
        compile_mode=cfg.compile_mode,
        use_weight_normalization=False,
    )

    # normalize network parameters after initialization
    actor.normalize_parameters()
    critic.normalize_parameters()
    target_critic.normalize_parameters()
    skill_encoder.normalize_parameters()

    return actor, critic, target_critic, temperature, skill_encoder, dual_lambda
