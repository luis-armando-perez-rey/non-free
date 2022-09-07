import os

import numpy as np
import torch
from torch import distributions as D
from models.distributions import MixtureDistribution

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


def get_encoding_distribution_constructor(encoder_distribution_type: str):
    """
    Get the function that returns the encoding distribution
    :param encoder_distribution_type: type of encoding distribution
    :return:
    """
    if encoder_distribution_type == "gaussian-mixture":
        def encoding_distribution_constructor(mean, logvar):
            device = mean.device
            mix = D.Categorical(torch.ones((mean.shape[0], mean.shape[1])).to(device))
            components = D.Independent(D.Normal(mean, torch.exp(logvar)), 1)
            distribution = D.MixtureSameFamily(mix, components)
            return distribution

    elif encoder_distribution_type == "von-mises-mixture":
        def encoding_distribution_constructor(mean, logvar):
            device = mean.device
            mix = D.Categorical(torch.ones((mean.shape[0], mean.shape[1])).to(device))
            angle = torch.atan2(mean[..., -2], mean[..., -1])
            components = D.von_mises.VonMises(loc=angle, concentration=1 / torch.exp(logvar[..., -1]))
            distribution = D.MixtureSameFamily(mix, components)
            return distribution
    else:
        encoding_distribution_constructor = None
        ValueError(f"{encoder_distribution_type} not available")
    return encoding_distribution_constructor


def cross_entropy_mixture(p1: torch.distributions.Distribution, p2: torch.distributions.Distribution,
                          n_samples: int = 20):
    """
    Estimates the crossentropy between two mixtures of distributions.
    :param p1: mixture of distributions 1
    :param p2: mixture of distributions 2
    :param n_samples: number of samples to estimate the crossentropy
    :return:
    """
    # Transform mean1 to angle
    sample1 = p1.sample((n_samples,))
    sample2 = p2.sample((n_samples,))

    return -p2.log_prob(sample1).sum(0).mean() - p1.log_prob(sample2).sum(0).mean()


def get_identity_loss_function(identity_loss_type: str, **kwargs):
    """
    Get the identity loss function. Identity loss function receives the encoded identity point for the current frame and
    the next frame.
    :param identity_loss_type: type of identity loss
    :return:
    """
    if identity_loss_type == "info-nce":
        def identity_loss_function(extra, extra_next):
            distance_matrix = (extra.unsqueeze(1) * extra_next.unsqueeze(0)).sum(-1) / kwargs["temperature"]
            loss = -torch.mean(
                (extra * extra_next).sum(-1) / kwargs["temperature"] - torch.logsumexp(distance_matrix, dim=-1))
            return loss

    elif identity_loss_type == "euclidean":
        def identity_loss_function(extra, extra_next):
            loss = torch.mean(torch.sum((extra - extra_next) ** 2, dim=-1))
            return loss
    else:
        identity_loss_function = None
        ValueError(f"{identity_loss_type} not available")
    return identity_loss_function


def get_z_values(p: MixtureDistribution, p_next: MixtureDistribution, extra: torch.tensor, extra_next: torch.tensor,
                 n_samples: int, autoencoder_type: str):
    if autoencoder_type == "ae":
        z = p.input_mean
        z_next = p_next.input_mean
    elif autoencoder_type == "vae":
        z = torch.movedim(p.sample_latent((n_samples,)), 0, 1)
        z_next = torch.movedim(p_next.sample_latent((n_samples,)), 0, 1)
    else:
        z = None
        z_next = None
        ValueError(f"Autoencoder type {autoencoder_type} not defined")
    # Do not change order! Append of extra dimension should be done after Von-Mises projection
    if extra.shape[-1] > 0:
        z = torch.cat([z, extra.unsqueeze(1).repeat(1, z.shape[1], 1)], dim=-1)
        z_next = torch.cat([z_next, extra_next.unsqueeze(1).repeat(1, z.shape[1], 1)], dim=-1)
    return z, z_next


# TODO: Remove after testing with equiv loss function
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


# TODO: Remove after testing with equiv loss function
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
