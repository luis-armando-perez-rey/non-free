import os
import numpy as np
import torch

pi = torch.tensor(np.pi)


def make_dir(path):
    os.makedirs(path, exist_ok=True)


def discrete_rot(z, angle, N):
    angle = torch.remainder(angle, 2 * pi * torch.ones(angle.shape).to(angle.device))
    disc = torch.arange(0, 2 * pi, step=2 * pi / N).to(z.device).unsqueeze(0)
    delta = torch.min(torch.abs(disc - angle), dim=-1)[1].unsqueeze(1)

    idx = torch.arange(N).view((1, N)).repeat((z.shape[0], 1)).to(z.device)
    idx_rolled = torch.remainder(idx + delta, N * torch.ones(idx.shape).to(z.device))
    redundant = z[:, idx_rolled.long()]
    return torch.diagonal(redundant, dim1=0, dim2=1).T


def rep_trick(mean, logvar):
    shape = mean.shape
    eps = torch.normal(torch.zeros(shape), torch.ones(shape)).to(mean.device)
    return eps * torch.exp(logvar / 2) + mean


def get_optimizer(optimizer_type: str, learning_rate: float, parameters):
    """
    Returns the optimizer
    :param optimizer_type: Type of optimizer to use
    :param learning_rate: Learning rate of the optimizer
    :param parameters: Parameters to optimize
    :return:
    """
    if optimizer_type == "adam":
        optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    elif optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=learning_rate)
    elif optimizer_type == "sgd":
        optimizer = torch.optim.SGD(parameters, lr=learning_rate, momentum=0.9, nesterov=True)
    else:
        ValueError(f"Optimizer {optimizer_type} not defined")
    return optimizer


def make_rotation_matrix(action):
    s = torch.sin(action)
    c = torch.cos(action)
    rot = torch.stack([torch.stack([c, -s]), torch.stack([s, c])])
    rot = rot.permute((2, 0, 1)).unsqueeze(1)
    return rot


def antisym_matrix(z):
    res = torch.zeros(z.shape[:-1] + (3, 3)).to(z.device)
    res[..., 0, 1] = z[..., 0]
    res[..., 0, 2] = z[..., 1]
    res[..., 1, 0] = -z[..., 0]
    res[..., 1, 2] = z[..., 2]
    res[..., 2, 0] = -z[..., 1]
    res[..., 2, 1] = -z[..., 2]
    return res
