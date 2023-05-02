"""
Resnet-18 in PyTorch
Inspired by https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
"""

import torch
from torch import nn
import torchvision
import torch.nn.functional as F


class ResizeConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        # noinspection PyTypeChecker
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x


class ResizeConv1d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        # noinspection PyTypeChecker
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x


class BasicBlockEnc(nn.Module):

    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = in_planes * stride

        # noinspection PyTypeChecker
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # noinspection PyTypeChecker
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if stride == 1:
            self.shortcut = nn.Sequential()
        else:
            # noinspection PyTypeChecker
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class BasicBlockEnc1D(nn.Module):

    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = in_planes * stride

        # Using a stride of 2 means that the input and output are downsampled by 2
        # noinspection PyTypeChecker
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        # noinspection PyTypeChecker
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)

        # Using a stride of 2 means that the input and output are downsampled by 2
        if stride == 1:
            self.shortcut = nn.Sequential()
        else:
            # noinspection PyTypeChecker
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class BasicBlockDec(nn.Module):

    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = int(in_planes / stride)

        # noinspection PyTypeChecker
        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)
        # self.bn1 could have been placed here, but that messes up the order of the layers when printing the class

        if stride == 1:
            # noinspection PyTypeChecker
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn2(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class BasicBlockDec1D(nn.Module):

    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = int(in_planes / stride)

        # noinspection PyTypeChecker
        self.conv2 = nn.Conv1d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(in_planes)
        # self.bn1 could have been placed here, but that messes up the order of the layers when printing the class

        if stride == 1:
            # noinspection PyTypeChecker
            self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm1d(planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv1d(in_planes, planes, kernel_size=3, scale_factor=stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential(
                ResizeConv1d(in_planes, planes, kernel_size=3, scale_factor=stride),
                nn.BatchNorm1d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn2(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class ResNet50Enc(nn.Module):
    """
    Loads the ResNet50 model and removes the last layer.
    """

    def __init__(self, z_dim: int = 10):
        super().__init__()
        self.z_dim = z_dim
        self.model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        self.model.fc = nn.Linear(2048, z_dim)

    def forward(self, x):
        x = self.model(x)
        return x


class ResNet18Enc(nn.Module):

    def __init__(self, num_blocks=None, z_dim=10, nc=3):
        super().__init__()
        if num_blocks is None:
            num_blocks = [2, 2, 2, 2]
        self.in_planes = 64
        self.z_dim = z_dim
        # noinspection PyTypeChecker
        self.conv1 = nn.Conv2d(nc, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(BasicBlockEnc, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(BasicBlockEnc, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(BasicBlockEnc, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(BasicBlockEnc, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512, z_dim)

    def _make_layer(self, block_class, planes, num_Blocks, stride):
        strides = [stride] + [1] * (num_Blocks - 1)
        layers = []
        for stride in strides:
            layers += [block_class(self.in_planes, stride)]
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


class ResNet50Dec(nn.Module):
    def __init__(self, nc=3, z_dim=10, num_blocks=None, ):
        super().__init__()
        if num_blocks is None:
            num_blocks = [2, 2, 2, 2, 2]
        self.in_planes = 2048
        self.z_dim = z_dim
        self.linear = nn.Linear(z_dim, self.in_planes)
        self.dim_activation = (self.in_planes, 7, 7)
        self.activation_size = self.dim_activation[0] * self.dim_activation[1] * self.dim_activation[2]
        self.reshape_linear = nn.Linear(self.in_planes, self.activation_size)  # shape (2048, 7, 7)
        self.layer1 = self._make_layer(BasicBlockDec, 1024, num_blocks[0], stride=2)  # shape (1024, 14, 14)
        self.layer2 = self._make_layer(BasicBlockDec, 512, num_blocks[1], stride=2)  # shape (512, 28, 28)
        self.layer3 = self._make_layer(BasicBlockDec, 256, num_blocks[2], stride=2)  # shape (256, 56, 56)
        self.layer4 = self._make_layer(BasicBlockDec, 128, num_blocks[3], stride=2)  # shape (128, 112, 112)
        self.layer5 = self._make_layer(BasicBlockDec, 64, num_blocks[4], stride=2)  # shape (64,224, 224)

        self.bn1 = nn.BatchNorm2d(64)
        # noinspection PyTypeChecker
        self.conv1 = nn.Conv2d(64, nc, kernel_size=3, stride=1, padding=1, bias=False)  # shape (3, 224, 224)

    def _make_layer(self, block_class, planes, num_Blocks, stride):
        strides = [stride] + [1] * (num_Blocks - 1)
        layers = []
        for stride in strides:
            layers += [block_class(self.in_planes, stride)]
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.relu(self.linear(x))
        x = self.reshape_linear(x)
        x = x.view(x.size(0), *self.dim_activation)
        # Apply inverse convolutional blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.bn1(x)
        x = self.conv1(x)
        return x


class ResNet18Dec(nn.Module):
    def __init__(self, nc=3, z_dim=10, num_blocks=None, ):
        super().__init__()
        if num_blocks is None:
            num_blocks = [2, 2, 2, 2]
        self.in_planes = 512
        self.z_dim = z_dim
        self.linear = nn.Linear(z_dim, 512)
        self.dim_activation = (512, 4, 4)
        self.activation_size = self.dim_activation[0] * self.dim_activation[1] * self.dim_activation[2]
        self.reshape_linear = nn.Linear(512, self.activation_size)  # shape (512, 4, 4)
        self.layer1 = self._make_layer(BasicBlockDec, 256, num_blocks[0], stride=2)  # shape (256, 8, 8)
        self.layer2 = self._make_layer(BasicBlockDec, 128, num_blocks[1], stride=2)  # shape (128, 16, 16)
        self.layer3 = self._make_layer(BasicBlockDec, 64, num_blocks[2], stride=2)  # shape (64, 32, 32)
        self.layer4 = self._make_layer(BasicBlockDec, 32, num_blocks[3], stride=2)  # shape (32, 64, 64)
        self.bn1 = nn.BatchNorm2d(32)
        # noinspection PyTypeChecker
        self.conv1 = nn.Conv2d(32, nc, kernel_size=3, stride=1, padding=1, bias=False)  # shape (3, 64, 64)

    def _make_layer(self, block_class, planes, num_Blocks, stride):
        strides = [stride] + [1] * (num_Blocks - 1)
        layers = []
        for stride in strides:
            layers += [block_class(self.in_planes, stride)]
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.relu(self.linear(x))
        x = self.reshape_linear(x)
        x = x.view(x.size(0), *self.dim_activation)
        # Apply inverse convolutional blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn1(x)
        x = self.conv1(x)
        return x


class EncMNIST(nn.Module):

    def __init__(self, latent_dim):
        super().__init__()

        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(True)
        )

        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        ### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(3 * 3 * 32, 128),
            nn.ReLU(True),
            nn.Linear(128, latent_dim)
        )

    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x


class DecMNIST(nn.Module):

    def __init__(self, encoded_space_dim):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 3 * 3 * 32),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1,
                                      unflattened_size=(32, 3, 3))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3,
                               stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2,
                               padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=2,
                               padding=1, output_padding=1)
        )

    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        return x


class ResNet1DEnc(nn.Module):
    def __init__(self, num_blocks=None, z_dim=10, nc=1):
        super().__init__()
        if num_blocks is None:
            num_blocks = [2, 2, 2]
        self.in_planes = 64
        self.z_dim = z_dim
        # noinspection PyTypeChecker
        self.conv1 = nn.Conv1d(nc, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.layer1 = self._make_layer(BasicBlockEnc1D, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(BasicBlockEnc1D, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(BasicBlockEnc1D, 256, num_blocks[2], stride=2)
        # self.layer4 = self._make_layer(BasicBlockEnc1D, 512, num_Blocks[3], stride=2)
        self.linear = nn.Linear(256, z_dim)

    def _make_layer(self, block_constructor, planes, num_Blocks, stride):
        strides = [stride] + [1] * (num_Blocks - 1)
        layers = []
        for stride in strides:
            layers += [block_constructor(self.in_planes, stride)]
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)
        x = F.adaptive_avg_pool1d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


class ResNet1DDec(nn.Module):
    def __init__(self, nc=1, z_dim=10, num_blocks=None):
        super().__init__()
        if num_blocks is None:
            num_blocks = [2, 2, 2]
        self.in_planes = 256
        self.z_dim = z_dim
        self.linear = nn.Linear(z_dim, 256)
        self.dim_activation = (256, 25)
        self.activation_size = self.dim_activation[0] * self.dim_activation[1]
        self.reshape_linear = nn.Linear(256, self.activation_size)
        self.layer1 = self._make_layer(BasicBlockDec1D, 256, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(BasicBlockDec1D, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(BasicBlockDec1D, 64, num_blocks[2], stride=1)
        self.bn1 = nn.BatchNorm1d(64)
        # noinspection PyTypeChecker
        self.conv1 = nn.Conv1d(64, nc, kernel_size=3, stride=2, padding=1, bias=False)

    def _make_layer(self, block_class, planes, num_Blocks, stride):
        strides = [stride] + [1] * (num_Blocks - 1)
        layers = []
        for stride in strides:
            layers += [block_class(self.in_planes, stride)]
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.relu(self.linear(x))
        x = self.reshape_linear(x)
        # Reshape back to 3D
        x = x.view(x.size(0), self.dim_activation)
        # Apply inverse convolutional blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.bn1(x)
        x = self.conv1(x)
        return x
