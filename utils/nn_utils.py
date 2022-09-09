import os

import numpy as np
import torch
from torch import distributions as D

pi = torch.tensor(np.pi)


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def discrete_rot(z, angle, N):
    angle = torch.remainder(angle, 2 * pi * torch.ones(angle.shape).to(angle.device))
    disc = torch.arange(0, 2 * pi, step=2 * pi / N).to(z.device).unsqueeze(0)
    delta = torch.min(torch.abs(disc - angle), dim=-1)[1].unsqueeze(1)

    idx = torch.arange(N).view((1, N)).repeat((z.shape[0], 1)).to(z.device)
    idx_rolled = torch.remainder(idx + delta, N * torch.ones(idx.shape).to(z.device))
    redundant = z[:, idx_rolled.long()]
    return torch.diagonal(redundant, dim1=0, dim2=1).T


def prob_loss(mean1, var1, mean2, var2, N):
    device = mean1.device
    mix1 = D.Categorical(torch.ones((mean1.shape[0], N)).to(device))
    comp1 = D.Independent(D.Normal(mean1, torch.exp(var1)), 1)
    gmm1 = D.MixtureSameFamily(mix1, comp1)

    mix2 = D.Categorical(torch.ones((mean1.shape[0], N)).to(device))
    comp2 = D.Independent(D.Normal(mean2, torch.exp(var2)), 1)
    gmm2 = D.MixtureSameFamily(mix2, comp2)

    sample1 = gmm1.sample((20,))
    sample2 = gmm1.sample((20,))

    return -gmm2.log_prob(sample1).sum(0).mean() - gmm1.log_prob(sample2).sum(0).mean()


def prob_loss_vm(mean1, logvar1, mean2, logvar2, N):
    """
    Estimates the crossentropy between two mixtures of N VonMises distributions. Receives mean values on the circle
    and converts them to angles. Then, it samples from the two distributions and estimates the crossentropy.
    Receives 2-d logvar of shape (batch_size, N, 2) and ignores one of the dimensions.
    Considers the concentration parameter as 1 / exp(logvar1[..., -1]) ignoring the second dimension of each logvar.
    :param mean1: mean1 on the circle
    :param logvar1: logvar1 of shape (batch_size, N, 2)
    :param mean2: mean2 on the circle
    :param logvar2: logvar2 of shape (batch_size, N, 2)
    :param N: number of components
    :return:
    """
    device = mean1.device
    # Transform mean1 to angle
    angle1 = torch.atan2(mean1[..., -2], mean1[..., -1])
    mix1 = D.Categorical(torch.ones((angle1.shape[0], N)).to(device))
    comp1 = D.von_mises.VonMises(loc=angle1, concentration=1 / torch.exp(logvar1[..., -1]))
    gmm1 = D.MixtureSameFamily(mix1, comp1)

    # Transform mean2 to angle
    angle2 = torch.atan2(mean2[..., -2], mean2[..., -1])
    mix2 = D.Categorical(torch.ones((angle2.shape[0], N)).to(device))

    comp2 = D.von_mises.VonMises(loc=angle2, concentration=1 / torch.exp(logvar2[..., -1]))
    gmm2 = D.MixtureSameFamily(mix2, comp2)
    sample1 = gmm1.sample((20,))
    sample2 = gmm1.sample((20,))

    return -gmm2.log_prob(sample1).sum(0).mean() - gmm1.log_prob(sample2).sum(0).mean()


def rep_trick(mean, logvar):
    shape = mean.shape
    eps = torch.normal(torch.zeros(shape), torch.ones(shape)).to(mean.device)
    return eps * torch.exp(logvar / 2) + mean

def antisym_matrix(z):
    res = torch.zeros(z.shape[:-1] + (3, 3)).to(z.device)
    res[...,0,1 ] = z[..., 0]
    res[...,0,2 ] = z[..., 1]
    res[...,1, 0 ] = -z[..., 0]
    res[...,1,2 ] = z[..., 2]
    res[...,2,0 ] = -z[..., 1]
    res[...,2,1 ] = -z[..., 2]
    return res
