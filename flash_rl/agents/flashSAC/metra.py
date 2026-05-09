from .agent import *
from .network import SkillEncoder

class METRAConfig(FlashSACConfig):
    pass


def _init_metra_networks(
    actor_observation_dim: int,
    critic_observation_dim: int,
    action_dim: int,
    skill_dim: int,
    skill_encoder_hidden_dim: int,
    skill_encoder_num_layers: int,
    cfg: FlashSACConfig,
    device: torch.device,
) -> tuple[Network, Network, Network, Network, Network]:
    # Create learning rate schedule
    warmup_cosine_decay_lr = warmup_cosine_decay_scheduler(
        init_value=cfg.learning_rate_init,
        peak_value=cfg.learning_rate_peak,
        end_value=cfg.learning_rate_end,
        warmup_steps=cfg.learning_rate_warmup_step,
        decay_steps=cfg.learning_rate_decay_step,
    )

    # Initialize actor
    actor_net = FlashSACActor(
        num_blocks=cfg.actor_num_blocks,
        input_dim=actor_observation_dim+skill_dim,
        hidden_dim=cfg.actor_hidden_dim,
        action_dim=action_dim,
    ).to(device)

    use_fused = device.type == "cuda" and torch.cuda.is_available()
    actor_optimizer = optim.Adam(actor_net.parameters(), lr=cfg.learning_rate_peak, fused=use_fused)
    actor_scheduler = torch.optim.lr_scheduler.LambdaLR(
        actor_optimizer,
        lr_lambda=lambda step: warmup_cosine_decay_lr(step) / cfg.learning_rate_peak,
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
        input_dim=critic_observation_dim + skill_dim + action_dim,
        hidden_dim=cfg.critic_hidden_dim,
        num_bins=cfg.critic_num_bins,
        min_v=cfg.critic_min_v,
        max_v=cfg.critic_max_v,
    ).to(device)

    critic_optimizer = optim.Adam(
        critic_net.parameters(),
        lr=cfg.learning_rate_peak,
        fused=use_fused,
    )
    critic_scheduler = torch.optim.lr_scheduler.LambdaLR(
        critic_optimizer,
        lr_lambda=lambda step: warmup_cosine_decay_lr(step) / cfg.learning_rate_peak,
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
        input_dim=critic_observation_dim + skill_dim + action_dim,
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
        lr=cfg.learning_rate_peak,
        fused=use_fused,
    )
    temp_scheduler = torch.optim.lr_scheduler.LambdaLR(
        temp_optimizer,
        lr_lambda=lambda step: warmup_cosine_decay_lr(step) / cfg.learning_rate_peak,
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
        obs_dim=critic_observation_dim,
        skill_dim=skill_dim,
        hidden_dim=skill_encoder_hidden_dim,
        num_layers=skill_encoder_num_layers,
    ).to(device)
    skill_encoder_optimizer = optim.Adam(
        skill_encoder_net.parameters(),
        lr=cfg.learning_rate_peak,
        fused=use_fused,
    )
    skill_encoder_scheduler = torch.optim.lr_scheduler.LambdaLR(
        skill_encoder_optimizer,
        lr_lambda=lambda step: warmup_cosine_decay_lr(step) / cfg.learning_rate_peak,
    )
    skill_encoder = Network(
        network=skill_encoder_net,
        optimizer=skill_encoder_optimizer,
        scheduler=skill_encoder_scheduler,
        compile_network=cfg.use_compile,
        compile_mode=cfg.compile_mode,
        use_weight_normalization=True,
    )

    # normalize network parameters after initialization
    actor.normalize_parameters()
    critic.normalize_parameters()
    target_critic.normalize_parameters()
    skill_encoder.normalize_parameters()

    return actor, critic, target_critic, temperature, skill_encoder

class METRAAgent(FlashSACAgent):
    def __init__(
            self, 
            observation_space, 
            action_space, 
            env_info, 
            cfg,
            skill_dim: int,
            skill_encoder_hidden_dim: int = 256,
            skill_encoder_num_layers: int = 2,
    ):
        super().__init__(observation_space, action_space, env_info, cfg)

        self._skill_dim = skill_dim
        self._skill_encoder_hidden_dim = skill_encoder_hidden_dim
        self._skill_encoder_num_layers = skill_encoder_num_layers

        # Observation dim change
        (
            self._actor,
            self._critic,
            self._target_critic,
            self._temperature,
            self._skill_encoder,
        ) = _init_metra_networks(
            actor_observation_dim=self._actor_observation_dim,
            critic_observation_dim=self._critic_observation_dim,
            action_dim=self._action_dim,
            skill_dim=self._skill_dim,
            skill_encoder_hidden_dim=self._skill_encoder_hidden_dim,
            skill_encoder_num_layers=self._skill_encoder_num_layers,
            cfg=self._cfg,
            device=self._device,
        )
        self._decoder = self._skill_encoder
