import torch
from .metra_config import METRAConfig
from .network import SkillEncoder, SkillGEncoder, SkillSimpleEncoder
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
skill_type      : str = 'continuous'
)               : 
    alignment, _ = compute_metra_intrinsic_reward(
        current_features, next_features, skills, skill_type
    )
    return alignment

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
            input_dim    = actor_observation_dim - skill_dim,
            output_dim   = skill_dim,
            hidden_sizes = [skill_encoder_hidden_dim] * skill_encoder_num_layers
        ).to(device)
    elif cfg.encoder_type  == "simple":
         skill_encoder_net  = SkillSimpleEncoder(
            obs_dim    = actor_observation_dim - skill_dim,
            skill_dim  = skill_dim,
            hidden_dim = skill_encoder_hidden_dim,
            num_layers = skill_encoder_num_layers,
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
)                 : 
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
    

def get_obs_normalizer_preset(normalizer_type: str) -> tuple[list[float], list[float]]:
    """Return (mean, std) lists for observation normalization presets.

    Precomputed from 10 000 length-50 random rollouts (without early termination),
    matching the official METRA ``iod/utils.py`` presets.
    """
    if normalizer_type == 'off':
        return [], []
    elif normalizer_type == 'ant_preset':
        mean = [
            0.00486117, 0.011312, 0.7022248, 0.8454677, -0.00102548, -0.00300276,
            0.00311523, -0.00139029, 0.8607109, -0.00185301, -0.8556998, 0.00343217,
            -0.8585605, -0.00109082, 0.8558013, 0.00278213, 0.00618173, -0.02584622,
            -0.00599026, -0.00379596, 0.00526138, -0.0059213, 0.27686235, 0.00512205,
            -0.27617684, -0.0033233, -0.2766923, 0.00268359, 0.27756855,
        ]
        std = [
            0.62473416, 0.61958003, 0.1717569, 0.28629342, 0.20020866, 0.20572574,
            0.34922406, 0.40098143, 0.3114514, 0.4024826, 0.31057045, 0.40343934,
            0.3110796, 0.40245822, 0.31100526, 0.81786263, 0.8166509, 0.9870919,
            1.7525449, 1.7468817, 1.8596431, 4.502961, 4.4070187, 4.522444,
            4.3518476, 4.5105968, 4.3704205, 4.5175962, 4.3704395,
        ]
        return mean, std
    elif normalizer_type == 'half_cheetah_preset':
        mean = [
            -0.07861924, -0.08627162, 0.08968642, 0.00960849, 0.02950368, -0.00948337,
            0.01661406, -0.05476654, -0.04932635, -0.08061652, -0.05205841, 0.04500197,
            0.02638421, -0.04570961, 0.03183838, 0.01736591, 0.0091929, -0.0115027,
        ]
        std = [
            0.4039283, 0.07610687, 0.23817, 0.2515473, 0.2698137, 0.26374814, 0.32229397,
            0.2896734, 0.2774097, 0.73060024, 0.77360505, 1.5871304, 5.5405455,
            6.7097645, 6.8253727, 6.3142195, 6.417641, 5.9759197,
        ]
        return mean, std
    else:
        raise ValueError(f"Unknown normalizer_type: {normalizer_type}")
