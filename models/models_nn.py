from lib2to3.pytree import Base
import torch
from torch import nn, load
from torch.nn import functional as F
from torchvision.utils import save_image
import torch.nn.functional as F
import torchvision.models as torch_models
from torch.autograd import Variable
import torchvision.transforms as transforms
import numpy as np
from functools import reduce
import utils.nn_utils as nn_utils
import ipdb
from models.resnet import *

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

class MDN(nn.Module):
    def __init__(self, nc, latent_dim, N, extra_dim=0):
        super().__init__()
        self.encoder = BaseEncoder(nc, 3*N + extra_dim)   #works only for latent_dim=2!
        self.latent_dim = latent_dim
        self.extra_dim = extra_dim
        self.N = N

    def forward(self, x):
        z = self.encoder(x)
        mean_pre = z[:,:self.N]
        mean = torch.cat( (torch.cos(mean_pre).unsqueeze(-1), torch.sin(mean_pre).unsqueeze(-1)), dim = -1)
        logvar = 5.4*torch.sigmoid(z[:, self.N : 3*self.N].view((-1,self.N,2))) -9.2
        extra = F.normalize(z[:,3*self.N: ], dim=-1)

        return mean, logvar, extra




class BaseEncoder(nn.Module):
    def __init__(self, nc, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(128, 128, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 1),
            nn.ReLU(True),
            View([-1, 256]),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )

    def forward(self, x):
        return self.encoder(x)



if __name__ == "__main__":
    pass