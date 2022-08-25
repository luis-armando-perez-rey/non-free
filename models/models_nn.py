from models.resnet import *


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class MDN(nn.Module):
    def __init__(self, nc, latent_dim, N, extra_dim=0, model='cnn'):
        super().__init__()
        if model == 'cnn':
            self.encoder = BaseEncoder(nc, 3 * N)  # works only for latent_dim=2!
            self.encoder_extra = BaseEncoder(nc, extra_dim)
        elif model == 'resnet':
            self.encoder = ResNet18Enc(z_dim=3 * N, nc=nc)  # works only for latent_dim=2!
            self.encoder_extra = ResNet18Enc(z_dim=extra_dim, nc=nc)
        self.latent_dim = latent_dim
        self.extra_dim = extra_dim
        self.N = N

    def forward(self, x):
        z = self.encoder(x)
        extra = F.normalize(self.encoder_extra(x), dim=-1)
        mean_pre = z[:, :self.N]
        mean = torch.cat((torch.cos(mean_pre).unsqueeze(-1), torch.sin(mean_pre).unsqueeze(-1)), dim=-1)
        logvar = 5.4 * torch.sigmoid(z[:, self.N: 3 * self.N].view((-1, self.N, 2))) - 9.2

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
