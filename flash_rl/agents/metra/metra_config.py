from dataclasses import dataclass

@dataclass
class METRAConfig:
    seed: int
    normalize_reward: bool
    normalized_G_max: float

    asymmetric_observation: bool
    device_type: str

    buffer_max_length: int
    buffer_min_length: int
    buffer_device_type: str
    sample_batch_size: int

    actor_learning_rate: float
    actor_num_blocks: int
    actor_hidden_dim: int
    actor_bc_alpha: float

    # action noise config when training
    actor_noise_zeta_mu: float
    actor_noise_zeta_max: int

    actor_update_period: int

    critic_learning_rate: float
    critic_num_blocks: int
    critic_hidden_dim: int
    critic_num_bins: int
    critic_min_v: float
    critic_max_v: float
    critic_target_update_tau: float

    temp_learning_rate: float
    temp_initial_value: float
    temp_target_sigma: float
    temp_target_entropy: float

    gamma: float
    n_step: int

    use_compile: bool
    compile_mode: str
    use_amp: bool

    load_optimizer: bool
    load_reward_normalizer: bool

    skill_dim: int
    skill_encoder_hidden_dim: int
    skill_encoder_num_layers: int
    skill_encoder_learning_rate: float
    skill_encoder_weight_norm: bool

    dual_lambda_learning_rate: float
    dual_lambda_init_value: float
    
    constraint_epsilon: float