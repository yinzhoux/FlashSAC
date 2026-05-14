import torch

def concat_obs_skill(
    observations: torch.Tensor,
    skills: torch.Tensor
) -> torch.Tensor:
    return torch.cat([observations, skills], dim=-1)

def sample_skills(num_envs: int, skill_dim: int, device) -> torch.Tensor:
    """
    Sample skill vector from unit circle.
    """
    skills = torch.randn((num_envs, skill_dim), device=device)
    return torch.nn.functional.normalize(skills, dim=-1)

@torch.compile
def sample_integer_from_cdf(cdf: torch.Tensor) -> torch.Tensor:
    """
    Sample an integer from the given CDF.
    Returns a 0-d int32 tensor.
    """
    u = torch.rand((), device=cdf.device)
    idx = torch.argmax((u < cdf).to(torch.int32))
    return (idx + 1).to(torch.int32)

@torch.compile
def build_truncated_zeta_cdf(mu: float, max_n: int) -> torch.Tensor:
    """
    Build the truncated zeta CDF for the given mu and max_n.
    """
    ns = torch.arange(1, max_n + 1, dtype=torch.float32)
    pmf = ns ** (-mu)
    pmf = pmf / torch.sum(pmf)
    cdf = torch.cumsum(pmf, dim=0)
    return cdf
