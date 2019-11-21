import numpy as np


def get_mean_params(gp):
    params = []
    for name, param in gp.named_parameters():
        if 'mean_function.mean_fn' in name:
            params.append(param)
    return params


def get_kernel_params(gp):
    """
    Exclude embedding parameters. NOTE tentative
    """
    params = []
    for name, param in gp.named_parameters():
        if 'mean_function.mean_fn' in name:
            continue
        params.append(param)
    return params
