import torch


def concat_obs_skill(observations: torch.Tensor, skills: torch.Tensor) -> torch.Tensor:
    return torch.cat([observations, skills], dim=-1)


def sample_skills(
    num_envs: int,
    skill_dim: int,
    device,
    skill_type: str = "continuous",
) -> torch.Tensor:
    """
    Sample skill vector from unit circle.
    """
    if skill_type == "discrete":
        skill_indices = torch.randint(0, skill_dim, (num_envs,), device=device)
        return torch.nn.functional.one_hot(skill_indices, num_classes=skill_dim).to(torch.float32)

    if skill_type != "continuous":
        raise ValueError(f"Unsupported skill_type: {skill_type}")

    skills = torch.randn((num_envs, skill_dim), device=device)
    return torch.nn.functional.normalize(skills, dim=-1)


def build_default_skills(
    num_envs: int,
    skill_dim: int,
    device,
    skill_type: str,
    default_skill_x: float,
    default_skill_y: float,
    default_skill_index: int,
) -> torch.Tensor:
    if skill_type == "discrete":
        if default_skill_index < 0 or default_skill_index >= skill_dim:
            raise ValueError(f"default_skill_index must be in [0, {skill_dim}), got {default_skill_index}")

        skill_indices = torch.full((num_envs,), default_skill_index, dtype=torch.long, device=device)
        return torch.nn.functional.one_hot(skill_indices, num_classes=skill_dim).to(torch.float32)

    if skill_type != "continuous":
        raise ValueError(f"Unsupported skill_type: {skill_type}")

    if skill_dim != 2:
        raise ValueError(f"Default continuous skill only supports skill_dim=2, got {skill_dim}")

    fixed_skill = torch.tensor(
        [default_skill_x, default_skill_y],
        dtype=torch.float32,
        device=device,
    )
    fixed_skill = torch.nn.functional.normalize(fixed_skill, dim=0, eps=1e-8)
    return fixed_skill.unsqueeze(0).expand(num_envs, -1).clone()


def build_metra_skill_masks(skills: torch.Tensor, skill_type: str) -> torch.Tensor:
    if skill_type == "continuous":
        return skills

    if skill_type != "discrete":
        raise ValueError(f"Unsupported skill_type: {skill_type}")

    skill_dim = skills.shape[-1]
    if skill_dim <= 1:
        raise ValueError(f"Discrete skill masks require skill_dim > 1, got {skill_dim}")

    mean_skill = skills.mean(dim=-1, keepdim=True)
    return (skills - mean_skill) * (skill_dim / (skill_dim - 1))


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
