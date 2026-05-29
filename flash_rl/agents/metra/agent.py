import copy
import numpy as np
import torch
from flash_rl.buffers.metra_buffer import METRATorchBuffer
from .config import METRAConfig
from .network import ParameterModule, build_option_policy, build_policy_module, build_q_functions
from flash_rl.agents.metra.garage import get_state_encoder
from gymnasium import Env

class METRAAgent:
    def __init__(self, env: Env, cfg: METRAConfig):
        self.cfg = cfg
        self.action_dim = env.action_space.shape[0]
        self.observ_dim = env.observation_space.shape[0]
        self.skill_dim = cfg.skill_dim

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hidden_sizes = [cfg.model_master_dim] * cfg.model_master_num_layers
        policy_q_input_dim = self.observ_dim + cfg.skill_dim

        policy_module = build_policy_module(
            input_dim=policy_q_input_dim,
            action_dim=self.action_dim,
            hidden_sizes=hidden_sizes,
            hidden_nonlinearity=torch.relu,
        )
        self.option_policy = build_option_policy(cfg.skill_dim, policy_module).to(self.device)

        qf1, qf2 = build_q_functions(
            obs_dim=policy_q_input_dim,
            action_dim=self.action_dim,
            hidden_sizes=hidden_sizes,
            hidden_nonlinearity=torch.relu,
        )
        self.qf1 = qf1.to(self.device)
        self.qf2 = qf2.to(self.device)
        self.target_qf1 = copy.deepcopy(qf1).to(self.device)
        self.target_qf2 = copy.deepcopy(qf2).to(self.device)

        self.skill_encoder = get_state_encoder(
            input_dim=self.observ_dim,
            output_dim=cfg.skill_dim,
            hidden_sizes=hidden_sizes,
            const_std=False,
            hidden_nonlinearity=torch.relu,
            w_init=torch.nn.init.xavier_uniform_,
            init_std=1,
            min_std=1e-6,
            max_std=None,
            spectral_normalization=cfg.spectral_normalization,
        ).to(self.device)

        self.dual_lam = ParameterModule(torch.tensor([np.log(cfg.dual_lam)], dtype=torch.float32)).to(self.device)
        self.log_alpha = ParameterModule(torch.tensor([np.log(cfg.alpha)], dtype=torch.float32)).to(self.device)

        self.optimizers = {
            "option_policy": torch.optim.Adam(self.option_policy.parameters(), lr=cfg.common_lr),
            "traj_encoder": torch.optim.Adam(self.skill_encoder.parameters(), lr=cfg.common_lr),
            "dual_lam": torch.optim.Adam(self.dual_lam.parameters(), lr=cfg.common_lr),
            "qf": torch.optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=cfg.common_lr),
            "log_alpha": torch.optim.Adam(self.log_alpha.parameters(), lr=cfg.common_lr),
        }

        self.replay_buffer = METRATorchBuffer(
            observation_space=env.observation_space,
            action_space=env.action_space,
            n_step=1,
            gamma=cfg.gamma,
            max_length=cfg.buffer_max_length,
            min_length=cfg.buffer_min_length,
            sample_batch_size=cfg.sample_batch_size,
            device_type='cpu',
            skill_dim=cfg.skill_dim,
        )

        self.target_entropy = -self.action_dim / 2.0 * cfg.target_coef
    
__all__ = [
    "METRAAgent"
]
    