import torch
from .metra_config import METRAConfig
from .network import SkillEncoder, SkillGEncoder
from flash_rl.agents.utils.network import Network
from flash_rl.agents.utils.function import build_metra_skill_masks
import torch.optim as optim
from flash_rl.agents.metra.network import (
    FlashSACActor,
    FlashSACDoubleCritic,
    FlashSACTemperature
)
from flash_rl.agents.metra.garage import get_state_encoder
from flash_rl.agents.utils.scheduler import warmup_cosine_decay_scheduler
def compute_metra_intrinsic_reward(
    current_features: torch.Tensor,
    next_features   : torch.Tensor,
    skills          : torch.Tensor,
    skill_type      : str = "continuous",
) -> tuple[torch.Tensor, torch.Tensor]:
    delta_features = next_features - current_features
    skill_masks = build_metra_skill_masks(skills, skill_type=skill_type)
    alignment = torch.sum(delta_features * skill_masks, dim=-1)
    squared_distance = (delta_features**2).mean(dim=-1)
    return alignment, squared_distance

def compute_metra_constraint(
  squared_distance: torch.Tensor,
  epsilon         : float,
) -> torch.Tensor : 
    return torch.minimum(
        torch.full_like(squared_distance, epsilon),
        1.0 - squared_distance,
    )

def compute_metra_reward(
current_features: torch.Tensor,
next_features   : torch.Tensor,
skills          : torch.Tensor,
epsilon         : float,
lambda_value    : float,
skill_type      : str = "continuous",
)               : 
    alignment, squared_distance = compute_metra_intrinsic_reward(
        current_features, next_features, skills, skill_type=skill_type
    )

    constraint_term = compute_metra_constraint(
        squared_distance, epsilon
    )

    return alignment + constraint_term * lambda_value

def init_metra_networks(
  actor_observation_dim                                         : int,
  critic_observation_dim                                        : int,
  action_dim                                                    : int,
  skill_dim                                                     : int,
  skill_encoder_hidden_dim                                      : int,
  skill_encoder_num_layers                                      : int,
  cfg                                                           : METRAConfig,
  device                                                        : torch.device,
) -> tuple[Network, Network, Network, Network, Network, Network]: 
      # ========================== Initialize actor ==========================
    actor_net = FlashSACActor(
        num_blocks = cfg.actor_num_blocks,
        input_dim  = actor_observation_dim,
        hidden_dim = cfg.actor_hidden_dim,
        action_dim = action_dim,
    ).to(device)

    use_fused = device.type == "cuda" and torch.cuda.is_available()

    actor_optimizer, actor_scheduler = gen_optimizer(
        parameters     = actor_net.parameters(),
        use_fused      = use_fused,
        scheduler_type = cfg.sac_scheduler_type,
        init_value     = cfg.sac_lr_init_value,
        peak           = cfg.sac_lr_peak,
        end_value      = cfg.sac_lr_end_value,
        warmup_steps   = cfg.sac_lr_warmup_steps,
        decay_steps    = cfg.sac_lr_decay_steps
    )

    actor = Network(
        network                  = actor_net,
        optimizer                = actor_optimizer,
        scheduler                = actor_scheduler,
        compile_network          = cfg.use_compile,
        compile_mode             = cfg.compile_mode,
        use_weight_normalization = True,
    )
      # Manually compile `get_mean_and_std` function
    if cfg.use_compile: 
        actor.network.get_mean_and_std = torch.compile(actor.network.get_mean_and_std, mode=cfg.compile_mode)  # type: ignore

      # ========================== Initialize critic ==========================
    critic_net = FlashSACDoubleCritic(
        num_blocks = cfg.critic_num_blocks,
        input_dim  = critic_observation_dim + action_dim,
        hidden_dim = cfg.critic_hidden_dim,
        num_bins   = cfg.critic_num_bins,
        min_v      = cfg.critic_min_v,
        max_v      = cfg.critic_max_v,
    ).to(device)

    critic_optimizer, critic_scheduler = gen_optimizer(
        parameters     = critic_net.parameters(),
        use_fused      = use_fused,
        scheduler_type = cfg.sac_scheduler_type,
        init_value     = cfg.sac_lr_init_value,
        peak           = cfg.sac_lr_peak,
        end_value      = cfg.sac_lr_end_value,
        warmup_steps   = cfg.sac_lr_warmup_steps,
        decay_steps    = cfg.sac_lr_decay_steps
    )
    
    critic = Network(
        network                  = critic_net,
        optimizer                = critic_optimizer,
        scheduler                = critic_scheduler,
        compile_network          = cfg.use_compile,
        compile_mode             = cfg.compile_mode,
        use_weight_normalization = True,
    )

      # ========================== Initialize target critic (same as critic but no optimizer) ==========================
    target_critic_net = FlashSACDoubleCritic(
        num_blocks = cfg.critic_num_blocks,
        input_dim  = critic_observation_dim + action_dim,
        hidden_dim = cfg.critic_hidden_dim,
        num_bins   = cfg.critic_num_bins,
        min_v      = cfg.critic_min_v,
        max_v      = cfg.critic_max_v,
    ).to(device)
    target_critic_net.load_state_dict(critic_net.state_dict())
    target_critic = Network(
        network                  = target_critic_net,
        optimizer                = None,
        scheduler                = None,
        compile_network          = cfg.use_compile,
        compile_mode             = cfg.compile_mode,
        use_weight_normalization = True,
        ema_source               = critic,                         # wire EMA update source
        ema_tau                  = cfg.critic_target_update_tau,
    )

      # ========================== Initialize temperature ==========================
    temp_net        = FlashSACTemperature(cfg.temp_initial_value).to(device)
    temp_optimizer, temp_scheduler = gen_optimizer(
        parameters     = temp_net.parameters(),
        use_fused      = use_fused,
        scheduler_type = cfg.sac_scheduler_type,
        init_value     = cfg.sac_lr_init_value,
        peak           = cfg.sac_lr_peak,
        end_value      = cfg.sac_lr_end_value,
        warmup_steps   = cfg.sac_lr_warmup_steps,
        decay_steps    = cfg.sac_lr_decay_steps
    )
    temperature = Network(
        network                  = temp_net,
        optimizer                = temp_optimizer,
        scheduler                = temp_scheduler,
        compile_network          = cfg.use_compile,
        compile_mode             = cfg.compile_mode,
        use_weight_normalization = False,
    )

      # ============================ Initialize encoder ============================

    if cfg.encoder_type  == "flash":
       skill_encoder_net  = SkillEncoder(
            obs_dim    = actor_observation_dim - skill_dim,
            skill_dim  = skill_dim,
            hidden_dim = skill_encoder_hidden_dim,
            num_layers = skill_encoder_num_layers,
        ).to(device)
    elif cfg.encoder_type  == "gaussian":
         skill_encoder_net  = SkillGEncoder(
            input_dim    = actor_observation_dim-skill_dim,
            output_dim   = skill_dim,
            hidden_sizes = [skill_encoder_hidden_dim] * skill_encoder_num_layers
        ).to(device)
    
    skill_encoder_optimizer, skill_encoder_scheduler = gen_optimizer(
        parameters     = skill_encoder_net.parameters(),
        use_fused      = use_fused,
        scheduler_type = cfg.encoder_scheduler_type,
        init_value     = cfg.encoder_lr_init_value,
        peak           = cfg.encoder_lr_peak,
        end_value      = cfg.encoder_lr_end_value,
        warmup_steps   = cfg.encoder_lr_warmup_steps,
        decay_steps    = cfg.encoder_lr_decay_steps
    )
    skill_encoder = Network(
        network                  = skill_encoder_net,
        optimizer                = skill_encoder_optimizer,
        scheduler                = skill_encoder_scheduler,
        compile_network          = cfg.use_compile,
        compile_mode             = cfg.compile_mode,
        use_weight_normalization = cfg.skill_encoder_weight_norm,
    )

    dual_lambda_net        = FlashSACTemperature(cfg.dual_lambda_init_value).to(device)
    dual_lambda_optimizer, dual_lambda_scheduler = gen_optimizer(
        parameters     = dual_lambda_net.parameters(),
        use_fused      = use_fused,
        scheduler_type = cfg.lambda_scheduler_type,
        init_value     = cfg.lambda_lr_init_value,
        peak           = cfg.lambda_lr_peak,
        end_value      = cfg.lambda_lr_end_value,
        warmup_steps   = cfg.lambda_lr_warmup_steps,
        decay_steps    = cfg.lambda_lr_decay_steps
    )
    dual_lambda = Network(
        network                  = dual_lambda_net,
        optimizer                = dual_lambda_optimizer,
        scheduler                = dual_lambda_scheduler,
        compile_network          = cfg.use_compile,
        compile_mode             = cfg.compile_mode,
        use_weight_normalization = False,
    )

      # normalize network parameters after initialization
    actor.normalize_parameters()
    critic.normalize_parameters()
    target_critic.normalize_parameters()

    if cfg.skill_encoder_weight_norm: 
        skill_encoder.normalize_parameters()

    return actor, critic, target_critic, temperature, skill_encoder, dual_lambda

def gen_optimizer(
    parameters,
    use_fused,

scheduler_type: str,
init_value    : float | None,
peak          : float,
end_value     : float | None,
warmup_steps  : int | None,
decay_steps   : int | None
)             : 
    if     scheduler_type  == 'cosine':
        assert init_value      != None
        assert end_value       != None
        assert warmup_steps    != None
        assert decay_steps     != None
        scale_scheduler  = warmup_cosine_decay_scheduler(
            init_value,
            peak,
            end_value,
            warmup_steps,
            decay_steps
        )
        optimizer = optim.Adam(
            params = parameters, lr = peak, fused = use_fused
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda = lambda step: scale_scheduler(step) / peak
        )
        return optimizer, scheduler
    
    elif scheduler_type == "constant":
        optimizer        = optim.Adam(parameters, lr=peak, fused=use_fused)
        
        scheduler = torch.optim.lr_scheduler.ConstantLR(
            optimizer = optimizer, factor = 1.0
        )

        return optimizer, scheduler
    
    else: 
        raise NotImplementedError