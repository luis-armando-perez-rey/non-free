import torch
import torch.nn.functional as F
import numpy as np
import os
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix, so3_log_map
import matplotlib.pyplot as plt
from scipy.linalg import logm
import ipdb
from torch import distributions as D

pi = torch.tensor(np.pi)


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def discrete_rot(z, angle, N):
    angle = torch.remainder(angle, 2*pi*torch.ones(angle.shape).to(angle.device))
    disc = torch.arange(0, 2*pi, step=2*pi / N).to(z.device).unsqueeze(0)
    delta = torch.min(torch.abs(disc - angle), dim=-1)[1].unsqueeze(1)

    idx = torch.arange(N).view((1, N)).repeat((z.shape[0], 1)).to(z.device)
    idx_rolled = torch.remainder(idx + delta, N * torch.ones(idx.shape).to(z.device))
    redundant = z[:, idx_rolled.long()]


    return torch.diagonal(redundant, dim1=0, dim2=1).T

def prob_loss(mean1, var1, mean2, var2, N):
    device = mean1.device
    mix1 = D.Categorical(torch.ones((mean1.shape[0],N)).to(device))
    comp1 = D.Independent(D.Normal(mean1, torch.exp(var1)), 1)
    gmm1 = D.MixtureSameFamily(mix1, comp1)

    mix2 = D.Categorical(torch.ones((mean1.shape[0],N)).to(device))
    comp2 = D.Independent(D.Normal(mean2, torch.exp(var2)), 1)
    gmm2 = D.MixtureSameFamily(mix2, comp2)

    sample1 = gmm1.sample((100,))
    sample2 = gmm1.sample((100,))


    return -gmm2.log_prob(sample1).sum(0).mean() -gmm1.log_prob(sample2).sum(0).mean()



def rep_trick(mean, logvar):
    shape = mean.shape
    eps = torch.normal(torch.zeros(shape), torch.ones(shape)).to(mean.device)
    return eps*torch.exp(logvar/2) + mean
