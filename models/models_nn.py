from models.resnet import *


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class MDN(nn.Module):
    def __init__(self, nc:int , latent_dim:int, N:int, extra_dim:int=0, model:str="cnn" normalize_extra:bool=True):
        super().__init__()
        if model == 'cnn':
            self.encoder = BaseEncoder(nc, 3 * N)  # works only for latent_dim=2!
            self.encoder_extra = BaseEncoder(nc, extra_dim)
        elif model == 'resnet':
            self.encoder = ResNet18Enc(z_dim=3 * N, nc=nc)  # works only for latent_dim=2!
            self.encoder_extra = ResNet18Enc(z_dim=extra_dim, nc=nc)
        self.latent_dim = latent_dim
        self.normalize_extra = normalize_extra
        self.extra_dim = extra_dim
        self.N = N

    def forward(self, x):
        z = self.encoder(x)
        extra = F.normalize(self.encoder_extra(x), dim=-1)
        mean_pre = z[:, :self.N]
        mean = torch.cat((torch.cos(mean_pre).unsqueeze(-1), torch.sin(mean_pre).unsqueeze(-1)), dim=-1)
        logvar = 5.4 * torch.sigmoid(z[:, self.N: 3 * self.N].view((-1, self.N, 2))) - 9.2
        if self.normalize_extra:
            extra = F.normalize(z[:, 3 * self.N:], dim=-1)
        else:
            extra = z[:, 3 * self.N:]

        return mean, logvar, extra


class BaseEncoder(nn.Module):
    def __init__(self, nc: int, latent_dim: int):
        """
        Based convolutional neural network encoder that takes images with nc channels and returns a latent vector of
        latent_dim dimensions.
        :param nc: number of input channels
        :param latent_dim: output latent dimension
        """
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
