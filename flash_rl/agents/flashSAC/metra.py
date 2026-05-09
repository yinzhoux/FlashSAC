from .agent import *
from .agent import _sample_flashsac_actions
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

        # Store original observation dims.
        self._raw_critic_observation_dim = self._critic_observation_dim
        self._raw_actor_observation_dim = self._actor_observation_dim
        self._skill_dim = skill_dim
        self._skill_encoder_hidden_dim = skill_encoder_hidden_dim
        self._skill_encoder_num_layers = skill_encoder_num_layers
        # skill-conditioned observation
        self._actor_observation_dim = self._raw_actor_observation_dim + self._skill_dim
        self._critic_observation_dim = self._raw_critic_observation_dim + self._skill_dim

        # Replace parent networks with skill-conditioned networks.
        (
            self._actor,
            self._critic,
            self._target_critic,
            self._temperature,
            self._skill_encoder,
        ) = _init_metra_networks(
            actor_observation_dim=self._raw_actor_observation_dim,
            critic_observation_dim=self._raw_critic_observation_dim,
            action_dim=self._action_dim,
            skill_dim=self._skill_dim,
            skill_encoder_hidden_dim=self._skill_encoder_hidden_dim,
            skill_encoder_num_layers=self._skill_encoder_num_layers,
            cfg=self._cfg,
            device=self._device,
        )

        # Rollout-time skill state. These are initialized lazily once the number
        # of vectorized environments is known. (Runtime state)
        self._train_skills: Optional[torch.Tensor] = None
        self._eval_skills: Optional[torch.Tensor] = None
        self._train_skill_resample_steps: Optional[torch.Tensor] = None

    def _sample_skills(self, num_envs: int) -> torch.Tensor:
        """
        Sample skill vector from unit circle.
        """
        skills = torch.randn((num_envs, self._skill_dim), device=self._device)
        return torch.nn.functional.normalize(skills, dim=-1)

    def _ensure_rollout_skill_state(self, num_envs: int, training: bool) -> torch.Tensor:
        """
        skill states initializer.
        """
        if training:
            if self._train_skills is None or self._train_skills.shape[0] != num_envs:
                self._train_skills = self._sample_skills(num_envs)
                self._train_skill_resample_steps = torch.zeros(num_envs, dtype=torch.int64, device=self._device)
            assert self._train_skills is not None
            return self._train_skills

        if self._eval_skills is None or self._eval_skills.shape[0] != num_envs:
            self._eval_skills = self._sample_skills(num_envs)
        return self._eval_skills

    def _augment_actor_observations(
        self,
        observations: torch.Tensor,
        skills: torch.Tensor,
    ) -> torch.Tensor:
        actor_observations = observations
        if self._cfg.asymmetric_observation:
            actor_observations = actor_observations[:, : self._raw_actor_observation_dim]
        return torch.cat([actor_observations, skills], dim=-1)

    def sample_actions(
        self,
        interaction_step: int,
        prev_transition: MutableMapping[str, Tensor],
        training: bool,
    ) -> Tensor:
        temperature = 1.0 if training else 0.0
        observations = torch.as_tensor(prev_transition["next_observation"], dtype=torch.float32).to(self._device)
        skills = self._ensure_rollout_skill_state(observations.shape[0], training=training)
        actor_observations = self._augment_actor_observations(observations, skills)

        with torch.no_grad():
            (
                self._cached_noise,
                actions,
                self._cur_noise_repeat_count,
                self._cur_noise_repeat_n,
            ) = _sample_flashsac_actions(
                actor=self._actor,
                noise=self._cached_noise,
                observations=actor_observations,
                temperature=temperature,
                cur_count=self._cur_noise_repeat_count,
                cur_n=self._cur_noise_repeat_n,
                zeta_cdf=self._zeta_cdf,
            )

        return actions.cpu().numpy()
