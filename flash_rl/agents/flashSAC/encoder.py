import torch
import torch.nn as nn
from torch.distributions import Normal, Independent


class GaussianSkillEncoder(nn.Module):
    """
    METRA-style trajectory encoder.

    This matches the official behavior at the interface level:
    - input: observations of shape [B, obs_dim]
    - output: an Independent Normal distribution over skill embeddings
    - dist.mean shape: [B, skill_dim]
    - dist.stddev shape: [B, skill_dim]

    Typical usage:
        dist = encoder(obs)
        z = dist.mean
        logp = dist.log_prob(target_skill)
    """

    def __init__(
        self,
        obs_dim: int,
        skill_dim: int,
        hidden_dim: int,
        num_layers: int,
        min_std: float = 1e-6,
        max_std: float | None = None,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        hidden_sizes = [hidden_dim] * num_layers

        self.mean_net = self._build_mlp(
            input_dim=obs_dim,
            output_dim=skill_dim,
            hidden_sizes=hidden_sizes,
        )
        self.log_std_net = self._build_mlp(
            input_dim=obs_dim,
            output_dim=skill_dim,
            hidden_sizes=hidden_sizes,
        )

        self.min_std = min_std
        self.max_std = max_std

        self._min_log_std = None if min_std is None else float(torch.log(torch.tensor(min_std)))
        self._max_log_std = None if max_std is None else float(torch.log(torch.tensor(max_std)))

    @staticmethod
    def _build_mlp(input_dim: int, output_dim: int, hidden_sizes: list[int]) -> nn.Sequential:
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_sizes:
            linear = nn.Linear(prev_dim, hidden_dim)
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        output = nn.Linear(prev_dim, output_dim)
        nn.init.xavier_uniform_(output.weight)
        nn.init.zeros_(output.bias)
        layers.append(output)
        return nn.Sequential(*layers)

    def forward(self, observations: torch.Tensor, training: bool | None = None):
        del training  # kept only for compatibility with your previous interface

        mean = self.mean_net(observations)
        log_std = self.log_std_net(observations)

        if self._min_log_std is not None or self._max_log_std is not None:
            log_std = torch.clamp(
                log_std,
                min=self._min_log_std,
                max=self._max_log_std,
            )

        std = torch.exp(log_std)

        # Match official traj_encoder behavior: return a diagonal Gaussian
        return Independent(Normal(mean, std), 1)

    def forward_mode(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Convenience helper matching the common 'use mean as embedding' pattern.
        """
        return self.forward(observations).mean