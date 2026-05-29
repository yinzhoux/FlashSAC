from dataclasses import dataclass


@dataclass
class METRAConfig:
    skill_dim: int = 2
    model_master_dim: int = 1024
    model_master_num_layers: int = 2

    spectral_normalization: bool = False

    dual_lam: float = 30
    alpha: float = 0.01
    common_lr: float = 1e-4
    gamma: float = 0.99

    buffer_max_length: int = 300000
    buffer_min_length: int = 10000
    sample_batch_size: int = 256

    target_coef: float = 1.0

    dual_slack: float = 1e-3
    discount: float = 0.99
    tau: float = 0.005