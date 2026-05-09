from .agent import *
from .agent import _sample_flashsac_actions
from .agent import _update_networks
from .network import SkillEncoder
from flash_rl.buffers.metra_buffer import METRATorchBuffer

@dataclass
class METRAConfig(FlashSACConfig):
    skill_dim: int
    skill_encoder_hidden_dim: int
    skill_encoder_num_layers: int
    skill_reward_scale: float
    constraint_epsilon: float
    dual_lambda_initial_value: float


def _compute_metra_intrinsic_reward(
    current_features: torch.Tensor,
    next_features: torch.Tensor,
    skills: torch.Tensor,
    reward_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    delta_features = next_features - current_features
    alignment = torch.sum(delta_features * skills, dim=-1)
    intrinsic_reward = reward_scale * alignment
    squared_distance = torch.sum(torch.square(current_features - next_features), dim=-1)
    return intrinsic_reward, alignment, squared_distance


def _compute_metra_constraint(
    squared_distance: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    return torch.minimum(
        torch.full_like(squared_distance, epsilon),
        1.0 - squared_distance,
    )


def _update_metra_skill_encoder(
    skill_encoder: Network,
    dual_lambda: Network,
    batch: dict[str, torch.Tensor],
    reward_scale: float,
    constraint_epsilon: float,
    device: torch.device,
    use_amp: bool,
    grad_scaler: Optional[GradScaler],
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    raw_observations = batch["raw_observation"]
    raw_next_observations = batch["raw_next_observation"]
    skills = torch.nn.functional.normalize(batch["skill"], dim=-1, eps=1e-8)

    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        current_features = skill_encoder(observations=raw_observations, training=True)
        next_features = skill_encoder(observations=raw_next_observations, training=True)

        # Clone compiled outputs before reuse to avoid overwritten CUDAGraph buffers.
        current_features = current_features.clone()
        next_features = next_features.clone()

        intrinsic_reward, alignment, squared_distance = _compute_metra_intrinsic_reward(
            current_features=current_features,
            next_features=next_features,
            skills=skills,
            reward_scale=reward_scale,
        )
        constraint_term = _compute_metra_constraint(
            squared_distance=squared_distance,
            epsilon=constraint_epsilon,
        )
        lambda_value = dual_lambda().detach()
        skill_encoder_objective = alignment + lambda_value * constraint_term
        skill_encoder_loss = -skill_encoder_objective.mean()

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
    skill_encoder.normalize_parameters()

    update_info = {
        "loss": skill_encoder_loss,
        "mean_objective": skill_encoder_objective.mean(),
        "mean_alignment": alignment.mean(),
        "mean_constraint": constraint_term.mean(),
        "mean_intrinsic_reward": intrinsic_reward.mean(),
        "mean_squared_distance": squared_distance.mean(),
        "lambda": lambda_value.mean(),
    }
    update_info = {f"skill_encoder/{key}": value for key, value in update_info.items()}
    return update_info, intrinsic_reward.detach(), constraint_term.detach()


def _update_metra_dual_lambda(
    dual_lambda: Network,
    constraint_term: torch.Tensor,
) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        constraint_term = constraint_term.detach()

    lambda_value = dual_lambda()
    dual_lambda_loss = (lambda_value * constraint_term).mean()

    assert dual_lambda.optimizer is not None
    dual_lambda.optimizer.zero_grad(set_to_none=True)
    dual_lambda_loss.backward()
    dual_lambda.optimizer.step()

    if dual_lambda.scheduler is not None:
        dual_lambda.scheduler.step()

    updated_lambda = dual_lambda().detach()
    return {
        "dual_lambda/loss": dual_lambda_loss.detach(),
        "dual_lambda/value": updated_lambda,
        "dual_lambda/mean_constraint": constraint_term.mean(),
    }


def _init_metra_networks(
    actor_observation_dim: int,
    critic_observation_dim: int,
    action_dim: int,
    skill_dim: int,
    skill_encoder_hidden_dim: int,
    skill_encoder_num_layers: int,
    cfg: METRAConfig,
    device: torch.device,
) -> tuple[Network, Network, Network, Network, Network, Network]:
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

    dual_lambda_net = FlashSACTemperature(cfg.dual_lambda_initial_value).to(device)
    dual_lambda_optimizer = optim.Adam(
        dual_lambda_net.parameters(),
        lr=cfg.learning_rate_peak,
        fused=use_fused,
    )
    dual_lambda_scheduler = torch.optim.lr_scheduler.LambdaLR(
        dual_lambda_optimizer,
        lr_lambda=lambda step: warmup_cosine_decay_lr(step) / cfg.learning_rate_peak,
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

class METRAAgent(FlashSACAgent):
    def __init__(
            self, 
            observation_space, 
            action_space, 
            env_info, 
            cfg: METRAConfig,
    ):
        super().__init__(observation_space, action_space, env_info, cfg)

        # Store original observation dims.
        self._raw_critic_observation_dim = self._critic_observation_dim
        self._raw_actor_observation_dim = self._actor_observation_dim
        self._skill_dim = cfg.skill_dim
        self._skill_encoder_hidden_dim = cfg.skill_encoder_hidden_dim
        self._skill_encoder_num_layers = cfg.skill_encoder_num_layers
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
            self._dual_lambda,
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

        # Use Metra buffer
        self._replay_buffer = METRATorchBuffer(
            observation_space=observation_space,
            action_space=action_space,
            n_step=self._cfg.n_step,
            gamma=self._cfg.gamma,
            max_length=self._cfg.buffer_max_length,
            min_length=self._cfg.buffer_min_length,
            sample_batch_size=self._cfg.sample_batch_size,
            device_type=self._cfg.buffer_device_type,
            skill_dim=self._skill_dim,
        )

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

    def _augment_critic_observations(
        self,
        observations: torch.Tensor,
        skills: torch.Tensor,
    ) -> torch.Tensor:
        critic_observations = observations[:, : self._raw_critic_observation_dim]
        return torch.cat([critic_observations, skills], dim=-1)

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

    def process_transition(self, transition: MutableMapping[str, Tensor]) -> None:
        observations = torch.as_tensor(transition["observation"], dtype=torch.float32, device=self._device)
        num_envs = observations.shape[0]
        skills = self._ensure_rollout_skill_state(num_envs, training=True)
        if self._train_skill_resample_steps is None or self._train_skill_resample_steps.shape[0] != num_envs:
            self._train_skill_resample_steps = torch.zeros(num_envs, dtype=torch.int64, device=self._device)
        assert self._train_skill_resample_steps is not None

        replay_transition = dict(transition)
        replay_transition["skill"] = skills.detach().cpu().numpy()
        replay_transition["skill_resample_step"] = self._train_skill_resample_steps.detach().cpu().numpy()
        self._replay_buffer.add(replay_transition)

        terminated = torch.as_tensor(transition["terminated"], dtype=torch.bool, device=self._device)
        truncated = torch.as_tensor(transition["truncated"], dtype=torch.bool, device=self._device)
        done = terminated | truncated

        self._train_skill_resample_steps += 1
        if torch.any(done):
            skills[done] = self._sample_skills(int(done.sum().item()))
            self._train_skill_resample_steps[done] = 0

    def update(self) -> dict[str, Any]:
        batch = cast(dict[str, torch.Tensor], self._replay_buffer.sample())

        for key, value in batch.items():
            batch[key] = value.to(self._device, non_blocking=True)

        batch["raw_observation"] = batch["observation"]
        batch["raw_next_observation"] = batch["next_observation"]
        skills = batch["skill"]
        raw_observations = batch["raw_observation"]
        raw_next_observations = batch["raw_next_observation"]

        skill_encoder_info, intrinsic_reward, constraint_term = _update_metra_skill_encoder(
            skill_encoder=self._skill_encoder,
            dual_lambda=self._dual_lambda,
            batch=batch,
            reward_scale=self._cfg.skill_reward_scale,
            constraint_epsilon=self._cfg.constraint_epsilon,
            device=self._device,
            use_amp=self._cfg.use_amp,
            grad_scaler=self._grad_scaler,
        )
        dual_lambda_info = _update_metra_dual_lambda(
            dual_lambda=self._dual_lambda,
            constraint_term=constraint_term,
        )

        batch["reward"] = intrinsic_reward
        batch["observation"] = self._augment_critic_observations(raw_observations, skills)
        batch["next_observation"] = self._augment_critic_observations(raw_next_observations, skills)
        batch["actor_observation"] = self._augment_actor_observations(raw_observations, skills)
        batch["actor_next_observation"] = self._augment_actor_observations(
            raw_next_observations,
            skills,
        )

        if self._cfg.normalize_reward:
            assert self.reward_normalizer is not None
            batch["reward"] = self.reward_normalizer.normalize_rewards(batch["reward"])

        _update_info = _update_networks(
            batch=batch,
            actor=self._actor,
            critic=self._critic,
            target_critic=self._target_critic,
            temperature=self._temperature,
            cfg=self._cfg,
            do_actor_update=(self._update_step % self._cfg.actor_update_period == 0),
            device=self._device,
            grad_scaler=self._grad_scaler,
        )
        self._update_step += 1

        update_info: dict[str, float] = {}
        for key, value in skill_encoder_info.items():
            if isinstance(value, torch.Tensor):
                update_info[key] = value.item()
            elif not isinstance(value, dict):
                update_info[key] = float(value)
        for key, value in dual_lambda_info.items():
            if isinstance(value, torch.Tensor):
                update_info[key] = value.item()
            elif not isinstance(value, dict):
                update_info[key] = float(value)
        for key, value in _update_info.items():
            if isinstance(value, torch.Tensor):
                update_info[key] = value.item()
            elif not isinstance(value, dict):
                update_info[key] = float(value)

        return update_info

    def save(self, path: str) -> None:
        super().save(path)
        self._skill_encoder.save(os.path.join(path, "skill_encoder.pt"))
        self._dual_lambda.save(os.path.join(path, "dual_lambda.pt"))

    def load(self, path: str) -> None:
        super().load(path)
        load_optimizer = self._cfg.load_optimizer
        self._skill_encoder.load(os.path.join(path, "skill_encoder.pt"), load_optimizer=load_optimizer)
        self._dual_lambda.load(os.path.join(path, "dual_lambda.pt"), load_optimizer=load_optimizer)
