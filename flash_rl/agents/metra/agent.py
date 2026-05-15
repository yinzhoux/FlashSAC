import os
from typing import Any, MutableMapping, Optional, cast
from dataclasses import replace
import math
import torch
from torch.amp.grad_scaler import GradScaler

from flash_rl.agents.base_agent import BaseAgent
from flash_rl.agents.utils.reward_normalization import RewardNormalizer
from flash_rl.types import Tensor

from .metra_config import METRAConfig
from .metra_comp import init_metra_networks
from flash_rl.buffers.metra_buffer import METRATorchBuffer
from flash_rl.agents.utils.function import *
from .update import (
    update_skill_encoder,
    update_dual_lambda,
    update_policy
)

from .metra_comp import compute_metra_reward

class METRAAgent(BaseAgent[METRAConfig]): 
    def __init__(
            self, 
            observation_space, 
            action_space, 
            env_info, 
    cfg: METRAConfig,
    )  : 
        super().__init__(observation_space, action_space, env_info, cfg)

        self._raw_observation_dim      = observation_space.shape[-1]
        self._skill_dim                = cfg.skill_dim
        self._skill_encoder_hidden_dim = cfg.skill_encoder_hidden_dim
        self._skill_encoder_num_layers = cfg.skill_encoder_num_layers
        self._observation_dim          = self._raw_observation_dim + self._skill_dim
        self._action_dim               = action_space.shape[-1]

        self._device        = torch.device(cfg.device_type)
        temp_target_entropy = 0.5 * self._action_dim * math.log(2 * math.pi * math.e * cfg.temp_target_sigma**2)
        self._cfg           = replace(self._cfg, temp_target_entropy=temp_target_entropy, compile_mode=cfg.compile_mode)
        
          # Network init
        (
            self._actor,
            self._critic,
            self._target_critic,
            self._temperature,
            self._skill_encoder,
            self._dual_lambda,
        ) = init_metra_networks(
            actor_observation_dim    = self._observation_dim,
            critic_observation_dim   = self._observation_dim,
            action_dim               = self._action_dim,
            skill_dim                = self._skill_dim,
            skill_encoder_hidden_dim = self._skill_encoder_hidden_dim,
            skill_encoder_num_layers = self._skill_encoder_num_layers,
            cfg                      = self._cfg,
            device                   = self._device,
        )

          # Rollout-time skill state. These are initialized lazily once the number
          # of vectorized environments is known. (Runtime state)
        self._train_skills              :                Optional[torch.Tensor] = None
        self._eval_skills               :                Optional[torch.Tensor] = None
        self._train_skill_resample_steps: Optional[torch.Tensor]                = None

        self._curr_noise            = torch.randn(self._action_space.shape, device=self._device)
        self._curr_noise_repeat_cnt = torch.tensor(0, dtype=torch.int32, device=self._device)
        self._curr_noise_repeat_n   = torch.tensor(1, dtype=torch.int32, device=self._device)
        
          # Use Metra buffer
        self._replay_buffer = METRATorchBuffer(
            observation_space = observation_space,
            action_space      = action_space,
            n_step            = self._cfg.n_step,
            gamma             = self._cfg.gamma,
            max_length        = self._cfg.buffer_max_length,
            min_length        = self._cfg.buffer_min_length,
            sample_batch_size = self._cfg.sample_batch_size,
            device_type       = self._cfg.buffer_device_type,
            skill_dim         = self._skill_dim,
        )

        self._zeta_cdf = build_truncated_zeta_cdf(
            self._cfg.actor_noise_zeta_mu, 
            self._cfg.actor_noise_zeta_max
        )

        self._grad_scaler = GradScaler(device=self._device.type, enabled=self._cfg.use_amp)
        self._update_step = 0

        self.reward_normalizer = None
        if self._cfg.normalize_reward and not self._cfg.use_encoder_to_update:
            self.reward_normalizer = RewardNormalizer(
                gamma=self._cfg.gamma,
                G_max=self._cfg.normalized_G_max,
                load_rms=self._cfg.load_reward_normalizer,
                device=self._device,
            )

    def set_eval_skills(self, skills: Tensor, normalize: bool = True) -> None: 
        """Set fixed evaluation skills used when ``training=False`` rollouts."""
        skill_tensor       = torch.as_tensor(skills, dtype=torch.float32, device=self._device)
        if skill_tensor.ndim == 1:
           skill_tensor       = skill_tensor.unsqueeze(0)
        if skill_tensor.ndim != 2:
            raise ValueError(
                f"Expected skills to have shape (num_envs, skill_dim) or (skill_dim,), got {tuple(skill_tensor.shape)}"
            )
        if skill_tensor.shape[-1] != self._skill_dim:
            raise ValueError(
                f"Skill dim mismatch: expected {self._skill_dim}, got {skill_tensor.shape[-1]}"
            )

        skill_tensor = torch.nn.functional.normalize(skill_tensor, dim=-1, eps=1e-8)
        
        self._eval_skills = skill_tensor.clone()

    def get_eval_skills(self) -> Optional[torch.Tensor]: 
        if  self._eval_skills is None                      : 
            return None
        return self._eval_skills.detach().clone()

    def get_or_make_skill(self, num_envs: int, training: bool) -> torch.Tensor: 
        """
        skill states initializer.
        """
        if self._cfg.use_encoder_to_update == False:
            if self._skill_dim != 2:
                raise ValueError(
                    f"Default fixed skill only supports skill_dim=2, got {self._skill_dim}"
                )

            fixed_skill = torch.tensor(
                [self._cfg.default_skill_x, self._cfg.default_skill_y],
                dtype=torch.float32,
                device=self._device,
            )
            fixed_skill = torch.nn.functional.normalize(fixed_skill, dim=0, eps=1e-8)
            fixed_skills = fixed_skill.unsqueeze(0).expand(num_envs, -1).clone()

            if training and (
                self._train_skill_resample_steps is None
                or self._train_skill_resample_steps.shape[0] != num_envs
            ):
                self._train_skill_resample_steps = torch.zeros(num_envs, dtype=torch.int64, device=self._device)

            return fixed_skills

        if training: 
            if self._train_skills is None or self._train_skills.shape[0] != num_envs: 
                self._train_skills               = sample_skills(num_envs, self._skill_dim, self._device)
                self._train_skill_resample_steps = torch.zeros(num_envs, dtype=torch.int64, device=self._device)
            assert self._train_skills is not None
            return self._train_skills

        if self._eval_skills is None or self._eval_skills.shape[0] != num_envs:
           self._eval_skills                                        = sample_skills(num_envs, self._skill_dim, self._device)
        return self._eval_skills

    def sample_actions(
        self,
      interaction_step: int,
      prev_transition : MutableMapping[str, Tensor],
      training        : bool,
    ) -> Tensor       : 
        temperature        = 1.0 if training else 0.0
        observations       = torch.as_tensor(prev_transition["next_observation"], dtype=torch.float32).to(self._device)
        skills             = self.get_or_make_skill(observations.shape[0], training=training)
        actor_observations = concat_obs_skill(observations, skills)

        with torch.no_grad(): 
            mean, std = self._actor.apply(
                "get_mean_and_std",
                observations = actor_observations,
                training     = False
            )

            if temperature == 0.0:
               actions      = torch.tanh(mean)
            else: 
                reinit    = (self._curr_noise_repeat_cnt == 0) | (self._curr_noise_repeat_cnt >= self._curr_noise_repeat_n)
                new_noise = torch.randn_like(mean)
                new_n     = sample_integer_from_cdf(self._zeta_cdf)

                  # update noise info if need
                self._curr_noise            = torch.where(reinit, new_noise, self._curr_noise)
                self._curr_noise_repeat_n   = torch.where(reinit, new_n, self._curr_noise_repeat_n)
                self._curr_noise_repeat_cnt = torch.where(reinit, torch.zeros_like(self._curr_noise_repeat_cnt), self._curr_noise_repeat_cnt)

                actions = torch.tanh(mean + std * self._curr_noise * temperature)

        return actions.cpu().numpy()

    def process_transition(self, transition: MutableMapping[str, Tensor]) -> None: 
        observations = torch.as_tensor(transition["observation"], dtype=torch.float32, device=self._device)
        num_envs     = observations.shape[0]
        skills       = self.get_or_make_skill(num_envs, training=True)
        assert self._train_skill_resample_steps is not None

        replay_transition                        = dict(transition)
        replay_transition["skill"]               = skills.detach().cpu().numpy()
        replay_transition["skill_resample_step"] = self._train_skill_resample_steps.detach().cpu().numpy()
        self._replay_buffer.add(replay_transition)

        if self._cfg.normalize_reward and not self._cfg.use_encoder_to_update:
            assert "reward" in transition and self.reward_normalizer is not None
            self.reward_normalizer.update_reward_stats(
                    reward=torch.as_tensor(transition["reward"], device=self._device),
                    terminated=torch.as_tensor(transition["terminated"], device=self._device),
                    truncated=torch.as_tensor(transition["truncated"], device=self._device),
            )

          # resample skills if done.
        terminated = torch.as_tensor(transition["terminated"], dtype=torch.bool, device=self._device)
        truncated  = torch.as_tensor(transition["truncated"], dtype=torch.bool, device=self._device)
        done       = terminated | truncated

        self._train_skill_resample_steps += 1
        if torch.any(done): 
            skills[done]                           = sample_skills(int(done.sum().item()), self._skill_dim, self._device)
            self._train_skill_resample_steps[done] = 0

    def update(self) -> dict[str, Any]: 
        assert self._cfg.temp_target_entropy != None
        batch  = cast(dict[str, torch.Tensor], self._replay_buffer.sample())

        for key, value in batch.items(): 
            batch[key] = value.to(self._device, non_blocking=True)

        batch["raw_observation"]      = batch["observation"]
        batch["raw_next_observation"] = batch["next_observation"]
        skills                        = batch["skill"]
        raw_observations              = batch["raw_observation"]
        raw_next_observations         = batch["raw_next_observation"]

        skill_encoder_info, _, constraint_term = update_skill_encoder(
            skill_encoder      = self._skill_encoder,
            batch              = batch,
            constraint_epsilon = self._cfg.constraint_epsilon,
            lambda_value       = self._dual_lambda().detach().clone(),
            device             = self._device,
            use_amp            = self._cfg.use_amp,
            grad_scaler        = self._grad_scaler,
        )
        dual_lambda_info = update_dual_lambda(
            dual_lambda     = self._dual_lambda,
            constraint_term = constraint_term,
        )

        with torch.no_grad(): 
            updated_current_features = self._skill_encoder(observations=raw_observations, training=False)
            updated_next_features    = self._skill_encoder(observations=raw_next_observations, training=False)

            updated_intrinsic_reward = compute_metra_reward(
                current_features = updated_current_features,
                next_features    = updated_next_features,
                skills           = skills,
                epsilon          = self._cfg.constraint_epsilon,
                lambda_value     = self._dual_lambda().detach().clone()
            )

        if self._cfg.use_encoder_to_update:
            batch["reward"] = updated_intrinsic_reward
        else:   # use env reward to update policy.
            pass

        # reward normalization.
        if self._cfg.normalize_reward:
            assert self.reward_normalizer is not None
            batch["reward"] = self.reward_normalizer.normalize_rewards(batch["reward"])

        batch["observation"]            = concat_obs_skill(raw_observations, skills)
        batch["next_observation"]       = concat_obs_skill(raw_next_observations, skills)
        batch["actor_observation"]      = concat_obs_skill(raw_observations, skills)
        batch["actor_next_observation"] = concat_obs_skill(
            raw_next_observations,
            skills,
        )

        assert self._cfg.temp_target_entropy != None
        _update_info                   = update_policy(
            batch           = batch,
            actor           = self._actor,
            critic          = self._critic,
            target_critic   = self._target_critic,
            temperature     = self._temperature,
            cfg             = self._cfg,
            do_actor_update = (self._update_step % self._cfg.actor_update_period == 0),
            device          = self._device,
            grad_scaler     = self._grad_scaler,
        )
        self._update_step += 1

        update_info                    : dict[str, float] = {}
        for key, value in skill_encoder_info.items(): 
            if  isinstance(value, torch.Tensor): 
                update_info[key] = value.item()
            elif not isinstance(value, dict): 
                update_info[key] = float(value)
        
        for key, value in dual_lambda_info.items(): 
            if  isinstance(value, torch.Tensor): 
                update_info[key] = value.item()
            elif not isinstance(value, dict): 
                update_info[key] = float(value)
        
        for key, value in _update_info.items(): 
            if  isinstance(value, torch.Tensor): 
                update_info[key] = value.item()
            elif not isinstance(value, dict): 
                update_info[key]          = float(value)
        
        update_info['env/reward'] = float(torch.mean(batch["reward"]).item())
        return update_info

    def save(self, path: str) -> None: 
        super().save(path)
        self._skill_encoder.save(os.path.join(path, "skill_encoder.pt"))
        self._dual_lambda.save(os.path.join(path, "dual_lambda.pt"))
        if self.reward_normalizer is not None:
            self.reward_normalizer.save(os.path.join(path, "reward_normalizer.pt"))

    def load(self, path: str) -> None: 
        super().load(path)
        load_optimizer = self._cfg.load_optimizer
        self._skill_encoder.load(os.path.join(path, "skill_encoder.pt"), load_optimizer=load_optimizer)
        self._dual_lambda.load(os.path.join(path, "dual_lambda.pt"), load_optimizer=load_optimizer)
        if self._cfg.load_reward_normalizer and self.reward_normalizer is not None:
            self.reward_normalizer.load(os.path.join(path, "reward_normalizer.pt"))

    def can_start_training(self): 
        return self._replay_buffer.can_sample()

    def get_metrics(self): 
        return {}
    
    def load_replay_buffer(self, path: str) -> None: 
        self._replay_buffer.load(os.path.join(path, "replay_buffer.pt"))
        print(f"\033[32m[FlashSAC]\033[0m Successfully loaded replay buffer from {path}.")

    def save_replay_buffer(self, path: str) -> None: 
        self._replay_buffer.save(os.path.join(path, "replay_buffer.pt"))
        print(f"\033[32m[FlashSAC]\033[0m Successfully saved replay buffer at {path}.")
