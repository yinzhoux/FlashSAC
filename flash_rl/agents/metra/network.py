import math

import torch
import torch.nn as nn

from flash_rl.agents.metra.layer import (
    EnsembleCategoricalValue,
    EnsembleFlashSACBlock,
    EnsembleFlashSACEmbedder,
    EnsembleUnitRMSNorm,
    FlashSACBlock,
    FlashSACEmbedder,
    NormalTanhPolicy,
    SkillEncoderBlock,
    UnitRMSNorm,
)

from .garage import get_state_encoder

class FlashSACActor(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        input_dim: int,
        hidden_dim: int,
        action_dim: int,
    ):
        super().__init__()
        self.embedder = FlashSACEmbedder(input_dim=input_dim, hidden_dim=hidden_dim)
        self.encoder = nn.ModuleList([FlashSACBlock(hidden_dim) for _ in range(num_blocks)])
        self.post_norm = UnitRMSNorm(hidden_dim)
        self.predictor = NormalTanhPolicy(hidden_dim=hidden_dim, action_dim=action_dim)

    def get_mean_and_std(
        self,
        observations: torch.Tensor,
        training: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = observations
        x = self.embedder(x, training)
        for block in self.encoder:
            x = block(x, training)
        x = self.post_norm(x)
        mean, std = self.predictor.get_mean_and_std(x, training)
        return mean, std

    def forward(
        self,
        observations: torch.Tensor,
        training: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = observations
        x = self.embedder(x, training)
        for block in self.encoder:
            x = block(x, training)
        x = self.post_norm(x)
        actions, info = self.predictor(x, training)
        return actions, info


class FlashSACDoubleCritic(nn.Module):
    """
    Double-Q for Clipped Double Q-learning.
    https://arxiv.org/pdf/1802.09477v3

    Fuses N parallel critic networks into single batched operations.
    All internal computation uses (N, batch, dim) tensor layout.
    """

    def __init__(
        self,
        num_blocks: int,
        input_dim: int,
        hidden_dim: int,
        num_bins: int,
        min_v: float,
        max_v: float,
        num_qs: int = 2,
    ):
        super().__init__()
        self.num_qs = num_qs

        self.embedder = EnsembleFlashSACEmbedder(num_qs, input_dim, hidden_dim)
        self.encoder = nn.ModuleList([EnsembleFlashSACBlock(num_qs, hidden_dim) for _ in range(num_blocks)])
        self.post_norm = EnsembleUnitRMSNorm(num_qs, hidden_dim)
        self.predictor = EnsembleCategoricalValue(
            num_ensemble=num_qs,
            hidden_dim=hidden_dim,
            num_bins=num_bins,
            min_v=min_v,
            max_v=max_v,
        )

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        training: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = torch.cat((observations, actions), dim=-1)  # [B, in_dim]
        x = x.unsqueeze(0).expand(self.num_qs, -1, -1)  # [num_qs, B, in_dim]
        x = self.embedder(x, training)
        for block in self.encoder:
            x = block(x, training)
        x = self.post_norm(x)
        qs, infos = self.predictor(x, training)
        return qs, infos


class FlashSACTemperature(nn.Module):
    def __init__(self, initial_value: float = 0.01):
        super().__init__()
        self.log_temp = nn.Parameter(torch.tensor([math.log(initial_value)], dtype=torch.float32))

    def forward(self) -> torch.Tensor:
        return torch.exp(self.log_temp)

class MetraEncoder(nn.Module):
    """Encode observations into a normalized skill embedding."""
    def __init__(
            self,
            obs_dim: int,
            skill_dim: int,
            hidden_dim: int,
            num_layers: int,
        ):
        super().__init__()
        self.encoder = SkillEncoderBlock(
            skill_dim=skill_dim,
            obs_dim=obs_dim,
            hidden_dim=hidden_dim,
            hidden_layers=num_layers,
        )

    def forward(
        self,
        observations: torch.Tensor,
        training: bool,
    ) -> torch.Tensor:
        return self.encoder(observations, training=training)


class MetraSimpleEncoder(nn.Module):
    """Simple MLP encoder matching the official METRA architecture.

    Plain feed-forward MLP — no residual blocks, no BatchNorm, no RMSNorm,
    no weight normalization.  Xavier-uniform init matches the official
    ``GaussianMLPIndependentStdModuleEx`` mean head.
    """

    def __init__(
        self,
        obs_dim: int,
        skill_dim: int,
        hidden_dim: int,
        num_layers: int,
        hidden_activation: str = "relu",
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        act: nn.Module = nn.ReLU() if hidden_activation == "relu" else nn.Tanh()

        layers: list[nn.Module] = []
        # First hidden layer
        layers.append(nn.Linear(obs_dim, hidden_dim))
        layers.append(act)
        # Additional hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(act)
        # Output layer
        layers.append(nn.Linear(hidden_dim, skill_dim))

        self.encoder = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.encoder.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        observations: torch.Tensor,
        training: bool,
    ) -> torch.Tensor:
        return self.encoder(observations)


class MetraGaussianEncoder(nn.Module):
    def __init__(self,
                input_dim,
                output_dim,
                hidden_sizes,
                hidden_nonlinearity=torch.relu,
                w_init=torch.nn.init.xavier_uniform_,
                init_std=1.0,
                min_std=1e-6,
                max_std=None,
                spectral_normalization=False):
        super().__init__()
        self.encoder = get_state_encoder(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_sizes=hidden_sizes,
            hidden_nonlinearity=hidden_nonlinearity,
            w_init=w_init,
            init_std=init_std,
            min_std=min_std,
            max_std=max_std,
            spectral_normalization=spectral_normalization   
        )

    def forward(self, observations: torch.Tensor, training: bool):
        return self.encoder(observations).mean

class SkillEncoder(MetraEncoder):
    """Backward-compatible alias for the METRA observation-to-skill encoder."""

    pass


class SkillSimpleEncoder(MetraSimpleEncoder):
    """Alias for the simple MLP encoder matching the official METRA architecture."""

    pass


class SkillGEncoder(MetraGaussianEncoder):
    pass
