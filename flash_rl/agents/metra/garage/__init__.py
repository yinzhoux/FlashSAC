import torch

from .gaussian_mlp_module_ex import *


def get_gaussian_module_construction(
    *,
    hidden_sizes,
    const_std=False,
    hidden_nonlinearity=torch.relu,
    w_init=torch.nn.init.xavier_uniform_,
    init_std=1.0,
    min_std=1e-6,
    max_std=None,
    spectral_normalization=False,
    **kwargs,
):
    module_kwargs = dict()

    if const_std:
        module_cls = GaussianMLPModuleEx
        module_kwargs.update(
            dict(
                learn_std=False,
                init_std=init_std,
            )
        )
    else:
        module_cls = GaussianMLPIndependentStdModuleEx
        module_kwargs.update(
            dict(
                std_hidden_sizes=hidden_sizes,
                std_hidden_nonlinearity=hidden_nonlinearity,
                std_hidden_w_init=w_init,
                std_output_w_init=w_init,
                init_std=init_std,
                min_std=min_std,
                max_std=max_std,
            )
        )

    module_kwargs.update(
        dict(
            hidden_sizes=hidden_sizes,
            hidden_nonlinearity=hidden_nonlinearity,
            hidden_w_init=w_init,
            output_w_init=w_init,
            std_parameterization="exp",
            bias=True,
            spectral_normalization=spectral_normalization,
            **kwargs,
        )
    )
    return module_cls, module_kwargs


def get_state_encoder(
    input_dim,
    output_dim,
    hidden_sizes,
    const_std=False,
    hidden_nonlinearity=torch.relu,
    w_init=torch.nn.init.xavier_uniform_,
    init_std=1.0,
    min_std=1e-6,
    max_std=None,
    spectral_normalization=False,
):
    module_cls, module_kwargs = get_gaussian_module_construction(
        hidden_sizes=hidden_sizes,
        const_std=const_std,
        hidden_nonlinearity=hidden_nonlinearity,
        w_init=w_init,
        init_std=init_std,
        min_std=min_std,
        max_std=max_std,
        spectral_normalization=spectral_normalization,
        input_dim=input_dim,
        output_dim=output_dim,
    )
    return module_cls(**module_kwargs)
