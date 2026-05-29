import math
from functools import partial
from typing import Optional

import numpy as np
import torch

from flash_rl.agents.metra.garage.gaussian_mlp_module_ex import (
    GaussianMLPTwoHeadedModuleEx,
    TanhNormal,
)
from flash_rl.agents.metra.garage.modules.mlp_module import MLPModule
from flash_rl.agents.metra.garage.policies.stochastic_policy import StochasticPolicy


def _calculate_fan_in_and_fan_out(tensor: torch.Tensor) -> tuple[int, int]:
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

    num_input_fmaps = tensor.size(1)
    num_output_fmaps = tensor.size(0)
    receptive_field_size = 1
    if tensor.dim() > 2:
        for size in tensor.shape[2:]:
            receptive_field_size *= size
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def _no_grad_normal_(
    tensor: torch.Tensor,
    mean: float,
    std: float,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    with torch.no_grad():
        return tensor.normal_(mean, std, generator=generator)


def xavier_normal_ex(tensor: torch.Tensor, gain: float = 1.0, multiplier: float = 0.1) -> torch.Tensor:
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    return _no_grad_normal_(tensor, 0.0, std * multiplier)


def build_policy_module(input_dim: int, action_dim: int, hidden_sizes: list[int], hidden_nonlinearity) -> torch.nn.Module:
    module_args = dict(
        hidden_sizes=hidden_sizes,
        layer_normalization=False,
        hidden_nonlinearity=hidden_nonlinearity,
        max_std=np.exp(2.0),
        normal_distribution_cls=TanhNormal,
        output_w_init=partial(xavier_normal_ex, gain=1.0),
        init_std=1.0,
    )
    return GaussianMLPTwoHeadedModuleEx(
        input_dim=input_dim,
        output_dim=action_dim,
        **module_args,
    )


def build_option_policy(skill_dim: int, module: torch.nn.Module) -> "PolicyEx":
    return PolicyEx(
        name="option_policy",
        option_info={"dim_option": skill_dim},
        module=module,
    )


def build_q_functions(obs_dim: int, action_dim: int, hidden_sizes: list[int], hidden_nonlinearity):
    qf1 = ContinuousMLPQFunctionEx(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=hidden_sizes,
        hidden_nonlinearity=hidden_nonlinearity or torch.relu,
    )
    qf2 = ContinuousMLPQFunctionEx(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=hidden_sizes,
        hidden_nonlinearity=hidden_nonlinearity or torch.relu,
    )
    return qf1, qf2


class PolicyEx(StochasticPolicy):
    def __init__(self,
                 name,
                 *,
                 module,
                 clip_action=False,
                 omit_obs_idxs=None,
                 option_info=None,
                 force_use_mode_actions=False,
                 ):
        super().__init__(env_spec=None, name=name)

        self._clip_action = clip_action
        self._omit_obs_idxs = omit_obs_idxs

        self._option_info = option_info
        self._force_use_mode_actions = force_use_mode_actions

        self._module = module

    def process_observations(self, observations):
        if self._omit_obs_idxs is not None:
            observations = observations.clone()
            observations[:, self._omit_obs_idxs] = 0
        return observations

    def forward(self, observations):
        observations = self.process_observations(observations)
        dist = self._module(observations)
        try:
            ret_mean = dist.mean
            ret_log_std = (dist.variance.sqrt()).log()
            info = dict(mean=ret_mean, log_std=ret_log_std)
        except NotImplementedError:
            info = dict()
        if hasattr(dist, '_normal'):
            info.update(dict(
                normal_mean=dist._normal.mean,
                normal_std=dist._normal.variance.sqrt(),
            ))

        return dist, info

    def forward_mode(self, observations):
        observations = self.process_observations(observations)
        samples = self._module.forward_mode(observations)
        return samples, dict()

    def forward_with_transform(self, observations, *, transform):
        observations = self.process_observations(observations)
        dist, dist_transformed = self._module.forward_with_transform(observations, transform=transform)
        try:
            ret_mean = dist.mean
            ret_log_std = (dist.variance.sqrt()).log()
            ret_mean_transformed = dist_transformed.mean.cpu()
            ret_log_std_transformed = (dist_transformed.variance.sqrt()).log().cpu()
            info = (dict(mean=ret_mean, log_std=ret_log_std),
                    dict(mean=ret_mean_transformed, log_std=ret_log_std_transformed))
        except NotImplementedError:
            info = (dict(),
                    dict())
        return (dist, dist_transformed), info

    def forward_with_chunks(self, observations, *, merge):
        observations = [self.process_observations(o) for o in observations]
        dist = self._module.forward_with_chunks(observations,
                                                merge=merge)
        try:
            ret_mean = dist.mean
            ret_log_std = (dist.variance.sqrt()).log()
            info = dict(mean=ret_mean, log_std=ret_log_std)
        except NotImplementedError:
            info = dict()

        return dist, info

    def get_mode_actions(self, observations):
        with torch.no_grad():
            if not isinstance(observations, torch.Tensor):
                observations = torch.as_tensor(observations).float().to(next(self.parameters()).device)
            samples, info = self.forward_mode(observations)
            return samples.cpu().numpy(), {
                k: v.detach().cpu().numpy()
                for (k, v) in info.items()
            }

    def get_sample_actions(self, observations):
        with torch.no_grad():
            if not isinstance(observations, torch.Tensor):
                observations = torch.as_tensor(observations).float().to(next(self.parameters()).device)
            dist, info = self.forward(observations)
            if isinstance(dist, TanhNormal):
                pre_tanh_values, actions = dist.rsample_with_pre_tanh_value()
                log_probs = dist.log_prob(actions, pre_tanh_values)
                actions = actions.detach().cpu().numpy()
                infos = {
                    k: v.detach().cpu().numpy()
                    for (k, v) in info.items()
                }
                infos['pre_tanh_value'] = pre_tanh_values.detach().cpu().numpy()
                infos['log_prob'] = log_probs.detach().cpu().numpy()
            else:
                actions = dist.sample()
                log_probs = dist.log_prob(actions)
                actions = actions.detach().cpu().numpy()
                infos = {
                    k: v.detach().cpu().numpy()
                    for (k, v) in info.items()
                }
                infos['log_prob'] = log_probs.detach().cpu().numpy()
            return actions, infos

    def get_actions(self, observations):
        assert isinstance(observations, np.ndarray) or isinstance(observations, torch.Tensor)
        if self._force_use_mode_actions:
            actions, info = self.get_mode_actions(observations)
        else:
            actions, info = self.get_sample_actions(observations)
        if self._clip_action:
            epsilon = 1e-6
            actions = np.clip(
                actions,
                self.env_spec.action_space.low + epsilon,
                self.env_spec.action_space.high - epsilon,
            )
        return actions, info

    def get_action(self, observation):
        with torch.no_grad():
            if not isinstance(observation, torch.Tensor):
                observation = torch.as_tensor(observation).float().to(next(self.parameters()).device)
            observation = observation.unsqueeze(0)
            action, agent_infos = self.get_actions(observation)
            return action[0], {k: v[0] for k, v in agent_infos.items()}


class ContinuousMLPQFunctionEx(MLPModule):
    def __init__(self, obs_dim, action_dim, **kwargs):
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        MLPModule.__init__(self, input_dim=self.obs_dim+self.action_dim, output_dim=1, **kwargs)

    def forward(self, observations, actions):
        return super().forward(torch.concat([observations, actions], 1))


class ParameterModule(torch.nn.Module):
    def __init__(self, init_value):
        super().__init__()

        self.param = torch.nn.Parameter(init_value)


__all__ = [
    "ContinuousMLPQFunctionEx",
    "ParameterModule",
    "PolicyEx",
    "build_option_policy",
    "build_policy_module",
    "build_q_functions",
    "xavier_normal_ex",
]
