from dataclasses import dataclass

@dataclass
class METRAConfig     : 
    seed            : int
    normalize_reward: bool
    normalized_G_max: float

    asymmetric_observation: bool
    device_type           : str

    buffer_max_length : int
    buffer_min_length : int
    buffer_device_type: str
    sample_batch_size : int

      # learning rate 
    sac_scheduler_type : str
    sac_lr_init_value  : float | None
    sac_lr_peak        : float
    sac_lr_end_value   : float | None
    sac_lr_warmup_rate : float
    sac_lr_warmup_steps: int | None
    sac_lr_decay_rate  : float
    sac_lr_decay_steps : int | None

    actor_num_blocks: int
    actor_hidden_dim: int
    actor_bc_alpha  : float

      # action noise config when training
    actor_noise_zeta_mu : float
    actor_noise_zeta_max: int

    actor_update_period: int

    critic_num_blocks       : int
    critic_hidden_dim       : int
    critic_num_bins         : int
    critic_min_v            : float
    critic_max_v            : float
    critic_target_update_tau: float

    temp_initial_value : float
    temp_target_sigma  : float
    temp_target_entropy: float

    gamma : float
    n_step: int

    use_compile : bool
    compile_mode: str
    use_amp     : bool

    load_optimizer        : bool
    load_reward_normalizer: bool


      # ==================== metra spec =======================
      # learning rate 
    encoder_scheduler_type : str
    encoder_lr_init_value  : float | None
    encoder_lr_peak        : float
    encoder_lr_end_value   : float | None
    encoder_lr_warmup_steps: int | None
    encoder_lr_warmup_rate : float
    encoder_lr_decay_rate  : float
    encoder_lr_decay_steps : int | None

    lambda_scheduler_type : str
    lambda_lr_init_value  : float | None
    lambda_lr_peak        : float
    lambda_lr_end_value   : float | None
    lambda_lr_warmup_steps: int | None
    lambda_lr_decay_steps : int | None

    encoder_type             : str
    skill_type               : str
    skill_dim                : int
    skill_encoder_hidden_dim : int
    skill_encoder_num_layers : int
    skill_encoder_weight_norm: bool

    dual_lambda_init_value: float
    constraint_epsilon    : float

    use_encoder_to_update: bool
    default_skill_x: float
    default_skill_y: float
    default_skill_index: int