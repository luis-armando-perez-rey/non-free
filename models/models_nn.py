from models.resnet import *


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class MDN(nn.Module):
    def __init__(self, nc: int, latent_dim: int, n_gaussians: int, extra_dim: int = 0, model: str = "cnn",
                 normalize_extra: bool = True):
        super().__init__()
        if model == 'cnn':
            self.encoder = BaseEncoder(nc, 3 * n_gaussians)  # works only for latent_dim=2!
            self.encoder_extra = BaseEncoder(nc, extra_dim)
        elif model == 'resnet':
            self.encoder = ResNet18Enc(z_dim=3 * n_gaussians, nc=nc)  # works only for latent_dim=2!
            self.encoder_extra = ResNet18Enc(z_dim=extra_dim, nc=nc)
        elif model == "dense":
            self.encoder = BaseDenseEncoder(latent_dim=3 * n_gaussians, nc=nc)  # works only for latent_dim=2!
            self.encoder_extra = BaseDenseEncoder(latent_dim=extra_dim, nc=nc)
        elif model == "cnn1":
            self.encoder = BaseEncoder1D(latent_dim=3 * n_gaussians, nc=nc)
            self.encoder_extra = BaseEncoder1D(latent_dim=extra_dim, nc=nc)
        self.latent_dim = latent_dim
        self.normalize_extra = normalize_extra
        self.extra_dim = extra_dim
        self.n_gaussians = n_gaussians

    def forward(self, x):
        z = self.encoder(x)
        mean_pre = z[:, :self.n_gaussians]
        mean = torch.cat((torch.cos(mean_pre).unsqueeze(-1), torch.sin(mean_pre).unsqueeze(-1)), dim=-1)
        logvar = 5.4 * torch.sigmoid(z[:, self.n_gaussians: 3 * self.n_gaussians].view((-1, self.n_gaussians, 2))) - 9.2
        if self.normalize_extra:
            extra = F.normalize(self.encoder_extra(x), dim=-1)
        else:
            extra = self.encoder_extra(x)

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

class BaseEncoder1D(nn.Module):
    def __init__(self, nc: int, latent_dim: int):
        """
        Based convolutional neural network encoder that takes images with nc channels and returns a latent vector of
        latent_dim dimensions.
        :param nc: number of input channels
        :param latent_dim: output latent dimension
        """
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Unflatten(1, (1, nc)),
            nn.Conv1d(1, 16, kernel_size=7, stride=1, padding="same"),
            nn.ReLU(),
            nn.Conv1d(16, 16, kernel_size=7, stride=1, padding="same"),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding="valid"),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, stride=1, padding="valid"),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.BatchNorm1d(736),
            nn.Linear(736, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(),
            nn.Linear(50, latent_dim),
        )

    def forward(self, x):
        return self.encoder(x)


class BaseDenseEncoder(nn.Module):
    def __init__(self, nc: int, latent_dim: int):
        """
        Based convolutional neural network encoder that takes images with nc channels and returns a latent vector of
        latent_dim dimensions.
        :param nc: number of input channels
        :param latent_dim: output latent dimension
        """
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(nc, 100),
            nn.ReLU(True),
            nn.Linear(100, 75),
            nn.ReLU(True),
            nn.Linear(75, 50),
            nn.ReLU(True),
            nn.Linear(50, 25),
            nn.ReLU(True),
            nn.Linear(25, 10),
            # nn.ReLU(True),
            # nn.Linear(10, 10),
            # nn.ReLU(True),
            # nn.Linear(10, 10),
            nn.ReLU(True),
            nn.Linear(10, latent_dim),
        )

    def forward(self, x):
        return self.encoder(x)


if __name__ == "__main__":
    pass
