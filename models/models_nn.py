import torch.nn.functional

from models.resnet import *
from utils.nn_utils import *
# from math import prod
from utils.nn_utils import *


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class Decoder(nn.Module):
    def __init__(self, nc: int, latent_dim: int, extra_dim: int = 0, model: str = "cnn"):
        super().__init__()
        total_latent_dim = latent_dim + extra_dim
        if model == "cnn1":
            self.decoder = BaseDecoder1D(nc, total_latent_dim)
        elif model == "dense":
            self.decoder = BaseDenseDecoder(nc, total_latent_dim)
        elif model == "resnet1d":
            self.decoder = ResNet1DDec(nc, total_latent_dim)
        elif model == "resnet":
            self.decoder = ResNet18Dec(nc, total_latent_dim)
        elif model == "mnistcnn":
            self.decoder = DecMNIST(encoded_space_dim=total_latent_dim)
        else:
            ValueError(f"{model} not available for decoder")

    def forward(self, x):
        return self.decoder(x)


class TorusEncode(nn.Module):
    """
    Module that outputs the location and scale parameters of a Gaussian distribution over Clifford Torus
    """

    def __init__(self, n_gaussians: int, latent_dim: int):
        super().__init__()
        self.n_gaussians = n_gaussians
        self.latent_dim = latent_dim

    def clifford_torus_projection(self, mean: torch.Tensor):
        """
        Normalize the dimensions pairwise to project the mean on the Clifford torus
        :param mean:
        :return:
        """
        device = mean.device
        projected_mean = torch.clone(mean)
        # Normalize the latent dimensions pairwise
        for num_pair in range(self.latent_dim // 2):
            projected_mean[:, :, num_pair * 2: (num_pair + 1) * 2] = torch.nn.functional.normalize(
                mean[:, :, num_pair * 2: (num_pair + 1) * 2], p=2, dim=-1)
        # Scale the embeddings
        # NOTE: Removed scale just for plotting purposes
        # projected_mean /= (1 / torch.sqrt(torch.tensor(self.latent_dim // 2, dtype=mean.dtype).to(device)))
        return projected_mean

    def forward(self, z):
        mean_pre = z[:, :self.latent_dim * self.n_gaussians].view((-1, self.n_gaussians, self.latent_dim))
        mean = self.clifford_torus_projection(mean_pre)
        logvar = 5.4 * torch.sigmoid(
            z[:, self.latent_dim * self.n_gaussians:].view((-1, self.n_gaussians, self.latent_dim))) - 9.2
        return mean, logvar


class SO3Encode(nn.Module):
    """
        Module that outputs the location and scale parameters of a 'Gaussian' distribution over SO(3)
    """

    def __init__(self, n_gaussians):
        super().__init__()
        self.n_gaussians = n_gaussians

    def forward(self, z):
        mean_pre = z[:, :3 * self.n_gaussians].view((-1, self.n_gaussians, 3))
        mean = torch.matrix_exp(antisym_matrix(mean_pre))
        logvar = 5.4 * torch.sigmoid(z[:, 3 * self.n_gaussians:].view((-1, self.n_gaussians, 3))) - 9.2
        return mean, logvar


class S1Encode(nn.Module):
    """
    Module that outputs the location and scale parameters of a Gaussian distribution over S^1
    """

    def __init__(self, n_gaussians):
        super().__init__()
        self.n_gaussians = n_gaussians

    def forward(self, z):
        mean_pre = z[:, :self.n_gaussians]
        mean = torch.cat((torch.cos(mean_pre).unsqueeze(-1), torch.sin(mean_pre).unsqueeze(-1)), dim=-1)
        logvar = 5.4 * torch.sigmoid(z[:, self.n_gaussians:].view((-1, self.n_gaussians, 2))) - 9.2
        return mean, logvar


class MDN(nn.Module):
    def __init__(self, nc: int, latent_dim: int, n_gaussians: int, extra_dim: int = 0, model: str = "cnn",
                 normalize_extra: bool = True):
        super().__init__()
        # Converts the latent dimension for the probabilistic output (means and covs of Gaussians)
        if latent_dim == 2:
            converted_dim = 3 * n_gaussians
        elif latent_dim == 3:
            converted_dim = 6 * n_gaussians
        elif latent_dim > 3 and latent_dim % 2 == 0:
            # 2 dimensions per Gaussian location parameter and 2 dimensions per Gaussian covariance parameter
            converted_dim = 2 * latent_dim * n_gaussians
        else:
            converted_dim = None
            ValueError(f"Latent dimension {latent_dim} not available for MDN")
        if model == 'cnn':
            self.encoder = BaseEncoder(nc, converted_dim)  # works only for latent_dim=2!
            self.encoder_extra = BaseEncoder(nc, extra_dim)
        elif model == 'resnet':
            self.encoder = ResNet18Enc(z_dim=converted_dim, nc=nc)  # works only for latent_dim=2!
            self.encoder_extra = ResNet18Enc(z_dim=extra_dim, nc=nc)
        elif model == "resnet1d":
            self.encoder = ResNet1DEnc(z_dim=converted_dim, nc=nc)
            self.encoder_extra = ResNet1DEnc(z_dim=extra_dim, nc=nc)
        elif model == "dense":
            self.encoder = BaseDenseEncoder(latent_dim=converted_dim, nc=nc)  # works only for latent_dim=2!
            self.encoder_extra = BaseDenseEncoder(latent_dim=extra_dim, nc=nc)
        elif model == "cnn1":
            self.encoder = BaseEncoder1D(latent_dim=converted_dim, nc=nc)
            self.encoder_extra = BaseEncoder1D(latent_dim=extra_dim, nc=nc)
        elif model == "dislib1":
            self.encoder = DisLibEncoder1D(latent_dim=converted_dim, nc=nc)
            self.encoder_extra = DisLibEncoder1D(latent_dim=extra_dim, nc=nc)
        elif model == "mnistcnn":
            self.encoder = EncMNIST(latent_dim = converted_dim)
            self.encoder_extra = EncMNIST(latent_dim = extra_dim)
        self.latent_dim = latent_dim
        self.normalize_extra = normalize_extra
        self.extra_dim = extra_dim
        self.n_gaussians = n_gaussians
        self.forward_function = self.set_forward_function()

    def set_forward_function(self):
        """
        Made this function to avoid if statements in the forward pass of the model (for speed)
        :return:
        """
        if self.latent_dim == 2:
            forward_function = S1Encode(self.n_gaussians)
        elif self.latent_dim == 3:
            forward_function = SO3Encode(self.n_gaussians)
        elif self.latent_dim > 3 and self.latent_dim % 2 == 0:
            forward_function = TorusEncode(self.n_gaussians, self.latent_dim)
        else:
            raise ValueError(f"Latent dimension {self.latent_dim} not available")
        return forward_function

    def forward(self, x):
        z = self.encoder(x)
        mean, logvar = self.forward_function(z)

        if self.normalize_extra:
            extra = F.normalize(self.encoder_extra(x), dim=-1)
        else:
            extra = self.encoder_extra(x)
        return mean, logvar, extra

class MDNSimplified(nn.Module):
    def __init__(self, nc: int, latent_dim: int, n_gaussians: int, extra_dim: int = 0, model: str = "cnn",
                 normalize_extra: bool = True):
        super().__init__()
        # Converts the latent dimension for the probabilistic output (means and covs of Gaussians)
        if latent_dim == 2:
            converted_dim = 3 * n_gaussians + extra_dim
        elif latent_dim == 3:
            converted_dim = 6 * n_gaussians + extra_dim
        elif latent_dim > 3 and latent_dim % 2 == 0:
            # 2 dimensions per Gaussian location parameter and 2 dimensions per Gaussian covariance parameter
            converted_dim = 2 * latent_dim * n_gaussians + extra_dim
        else:
            converted_dim = None
            ValueError(f"Latent dimension {latent_dim} not available for MDN")
        if model == 'cnn':
            self.encoder = BaseEncoder(nc, converted_dim)  # works only for latent_dim=2!
        elif model == 'resnet':
            self.encoder = ResNet18Enc(z_dim=converted_dim, nc=nc)  # works only for latent_dim=2!
        elif model == "resnet1d":
            self.encoder = ResNet1DEnc(z_dim=converted_dim, nc=nc)
        elif model == "dense":
            self.encoder = BaseDenseEncoder(latent_dim=converted_dim, nc=nc)  # works only for latent_dim=2!
        elif model == "cnn1":
            self.encoder = BaseEncoder1D(latent_dim=converted_dim, nc=nc)
        elif model == "dislib1":
            self.encoder = DisLibEncoder1D(latent_dim=converted_dim, nc=nc)
        elif model == "mnistcnn":
            self.encoder = EncMNIST(latent_dim = converted_dim)
        self.latent_dim = latent_dim
        self.normalize_extra = normalize_extra
        self.extra_dim = extra_dim
        self.n_gaussians = n_gaussians
        self.forward_function = self.set_forward_function()

    def set_forward_function(self):
        """
        Made this function to avoid if statements in the forward pass of the model (for speed)
        :return:
        """
        if self.latent_dim == 2:
            forward_function = S1Encode(self.n_gaussians)
        elif self.latent_dim == 3:
            forward_function = SO3Encode(self.n_gaussians)
        elif self.latent_dim > 3 and self.latent_dim % 2 == 0:
            forward_function = TorusEncode(self.n_gaussians, self.latent_dim)
        else:
            raise ValueError(f"Latent dimension {self.latent_dim} not available")
        return forward_function

    def forward(self, x):
        z = self.encoder(x)
        if self.extra_dim == 0:
            mean, logvar = self.forward_function(z)
            extra = None
        else:
            mean, logvar = self.forward_function(z[:, :-self.extra_dim])
            extra = z[:, -self.extra_dim:]
            if self.normalize_extra:
                extra = F.normalize(extra, dim=-1)
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


class DisLibEncoder1D(nn.Module):
    def __init__(self, nc: int, latent_dim: int):
        """
        Based convolutional neural network encoder that takes images with nc channels and returns a latent vector of
        latent_dim dimensions.
        :param nc: number of input channels
        :param latent_dim: output latent dimension
        """
        super().__init__()
        # noinspection PyTypeChecker
        self.encoder = nn.Sequential(
            nn.Unflatten(1, (1, nc)),
            nn.Conv1d(1, 16, kernel_size=4, stride=2, padding="valid"),
            nn.ReLU(),
            nn.Conv1d(16, 16, kernel_size=4, stride=2, padding="valid"),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=4, stride=2, padding="valid"),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=4, stride=2, padding="valid"),
            nn.ReLU(),
            nn.Flatten(),
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
        # noinspection PyTypeChecker
        self.encoder = nn.Sequential(
            nn.Unflatten(1, (1, nc)),
            nn.Conv1d(1, 16, kernel_size=7, stride=1, padding="same"),
            nn.ReLU(),
            nn.Conv1d(16, 16, kernel_size=7, stride=1, padding="same"),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Flatten(),
            # nn.BatchNorm1d(640),
            nn.Linear(800, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(),
            nn.Linear(50, latent_dim),
        )

    def forward(self, x):
        return self.encoder(x)


class BaseDecoder1D(nn.Module):
    def __init__(self, nc: int, dim_latent_extra: int):
        """
        Based convolutional neural network encoder that takes images with nc channels and returns a latent vector of
        latent_dim dimensions.
        :param nc: number of input channels
        :param latent_dim: output latent dimension
        """
        super().__init__()

        activation_shape = (32, 2 * 25)
        max_pool_size = activation_shape[0] * activation_shape[1]
        # noinspection PyTypeChecker
        self.encoder = nn.Sequential(
            nn.Linear(dim_latent_extra, 50),
            nn.ReLU(),
            nn.Linear(50, 100),
            nn.ReLU(),
            nn.Linear(100, 800),
            nn.ReLU(),
            nn.Linear(800, max_pool_size),
            nn.ReLU(),
            nn.Unflatten(-1, activation_shape),
            nn.Conv1d(32, 32, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(16, 16, kernel_size=7, stride=1, padding="same"),
            nn.ReLU(),
            nn.Conv1d(16, 1, kernel_size=7, stride=1, padding="same"),
            Squeeze(),
        )

    def forward(self, x):
        return self.encoder(x)


class Squeeze(torch.nn.Module):
    def forward(self, x):
        return torch.squeeze(x)


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
            nn.ReLU(True),
            nn.Linear(10, latent_dim),
        )

    def forward(self, x):
        return self.encoder(x)


class BaseDenseDecoder(nn.Module):
    def __init__(self, nc: int, dim_latent_extra: int):
        """
        Based convolutional neural network encoder that takes images with nc channels and returns a latent vector of
        latent_dim dimensions.
        :param nc: number of input channels
        :param latent_dim: output latent dimension
        """
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(dim_latent_extra, 10),
            nn.ReLU(),
            nn.Linear(10, 25),
            nn.ReLU(),
            nn.Linear(25, 50),
            nn.ReLU(),
            nn.Linear(50, 75),
            nn.ReLU(),
            nn.Linear(75, 100),
            nn.ReLU(),
            nn.Linear(100, nc)
        )

    def forward(self, x):
        return self.decoder(x)


if __name__ == "__main__":
    pass
