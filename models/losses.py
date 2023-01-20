import torch
import numpy as np
from models.distributions import MixtureDistribution


def matrix_dist(z_mean_next, z_mean_pred):
    latent_dim = z_mean_next.shape[-1]
    if latent_dim == 3:
        return ((z_mean_next.unsqueeze(2) - z_mean_pred.unsqueeze(1)) ** 2).sum(-1).sum(-1)
    else:
        return ((z_mean_next.unsqueeze(2) - z_mean_pred.unsqueeze(1)) ** 2).sum(-1)



class EquivarianceLoss:
    def __init__(self, loss_type: str, **kwargs):
        self.kwargs = kwargs
        self.loss_function = loss_type

    @property
    def loss_function(self):
        return self._loss_function

    @loss_function.setter
    def loss_function(self, loss_type: str):
        """
        Get the equivariance loss function. Equivariance loss function receives the encoding distribution for the
        current frame and the next frame.
        :param loss_type: type of equivariance loss
        :return:
        """
        if loss_type == "cross-entropy":
            equivariance_loss_function = self.cross_entropy_mixture
        elif loss_type == "chamfer_val":
            def equivariance_loss_function(p: MixtureDistribution,
                                           p_next: MixtureDistribution):
                mean = p.input_mean
                mean_next = p_next.input_mean
                loss = matrix_dist(mean, mean_next).min(dim=-1)[0].sum(dim=-1).mean()
                return loss
        elif loss_type == "chamfer":
            def equivariance_loss_function(p: MixtureDistribution,
                                           p_next: MixtureDistribution):
                mean = p.input_mean
                mean_next = p_next.input_mean
                if "chamfer_reg" not in self.kwargs:
                    chamfer_reg = 0.001
                else:
                    chamfer_reg = self.kwargs["chamfer_reg"]

                # loss = ((mean.unsqueeze(1) - mean_next.unsqueeze(2)) ** 2).sum(-1).min(dim=-1)[0].sum(
                #     dim=-1).mean()

                loss = matrix_dist(mean, mean_next).min(dim=-1)[0].sum(dim=-1).mean()
                # reg = ((mean.unsqueeze(1) - mean.unsqueeze(2)) ** 2).sum(-1).mean()
                reg = matrix_dist(mean, mean).mean()
                loss += chamfer_reg * reg
                return loss
        elif loss_type == "chamfer2":
            def equivariance_loss_function(p: MixtureDistribution,
                                           p_next: MixtureDistribution):
                mean = p.input_mean
                mean_next = p_next.input_mean
                if "chamfer_reg" not in self.kwargs:
                    chamfer_reg = 0.001
                else:
                    chamfer_reg = self.kwargs["chamfer_reg"]

                # loss = ((mean.unsqueeze(1) - mean_next.unsqueeze(2)) ** 2).sum(-1).min(dim=-1)[0].sum(
                #     dim=-1).mean()
                loss = matrix_dist(mean, mean_next).min(dim=-1)[0].mean()
                # reg = ((mean.unsqueeze(1) - mean.unsqueeze(2)) ** 2).sum(-1).mean()
                reg = matrix_dist(mean, mean).mean()
                loss += chamfer_reg * reg
                return loss
        elif loss_type == "euclidean":
            def equivariance_loss_function(p: MixtureDistribution,
                                           p_next: MixtureDistribution):
                mean = p.input_mean
                mean_next = p_next.input_mean
                loss = ((mean - mean_next) ** 2).sum(-1).mean()
                return loss
        else:
            equivariance_loss_function = None
            ValueError(f"{loss_type} not available")
        self._loss_function = equivariance_loss_function

    @staticmethod
    def get_location_parameter_function(encoder_distribution_type: str):
        """
        Get the function that returns the location parameter of the encoding distribution. This is useful when using
        mixture distributions.
        :param encoder_distribution_type: type of encoding distribution
        :return:
        """
        if encoder_distribution_type == "gaussian-mixture":
            def location_parameter_function(p):
                return p.component_distribution.base_dist.mean
        elif encoder_distribution_type == "von-mises-mixture":
            def location_parameter_function(p):
                return p.component_distribution.mean
        else:
            location_parameter_function = None
            ValueError(f"{encoder_distribution_type} not available")
        return location_parameter_function

    def cross_entropy_mixture(self, p1, p2: torch.distributions.Distribution):
        """
            Estimates the crossentropy between two mixtures of distributions.
            :param p1: mixture of distributions 1
            :param p2: mixture of distributions 2
            :param n_samples: number of samples to estimate the crossentropy
            :return:
            """
        if "n_samples" in self.kwargs:
            n_samples = self.kwargs["n_samples"]
        else:
            n_samples = 20
        # Transform mean1 to angle
        sample1 = p1.sample((n_samples,))
        sample2 = p2.sample((n_samples,))

        return -p2.log_prob(sample1).sum(0).mean() - p1.log_prob(sample2).sum(0).mean()

    def __call__(self, p, p_next):
        return self.loss_function(p, p_next)


class IdentityLoss:
    def __init__(self, loss_type: str, **kwargs):
        self.kwargs = kwargs
        self.loss_function = loss_type

    @property
    def loss_function(self):
        return self._loss_function

    @loss_function.setter
    def loss_function(self, loss_type: str):
        """
            Get the identity loss function. Identity loss function receives the encoded identity point for the current
            frame and the next frame.
            :param loss_type: type of identity loss
            :return:
            """
        if loss_type == "infonce":
            def identity_loss_function(extra, extra_next):
                distance_matrix = (extra.unsqueeze(1) * extra_next.unsqueeze(0)).sum(-1) / self.kwargs["temperature"]
                loss = -torch.mean(
                    (extra * extra_next).sum(-1) / self.kwargs["temperature"] - torch.logsumexp(distance_matrix,
                                                                                                dim=-1))
                return loss

        elif loss_type == "euclidean":
            def identity_loss_function(extra, extra_next):
                loss = torch.mean(torch.sum((extra - extra_next) ** 2, dim=-1))
                return loss
        else:
            identity_loss_function = None
            ValueError(f"{loss_type} not available")
        self._loss_function = identity_loss_function

    def __call__(self, extra, extra_next):
        return self.loss_function(extra, extra_next)


class ReconstructionLoss:
    def __init__(self, loss_type: str, **kwargs):
        self.kwargs = kwargs
        self.loss_function = loss_type

    @property
    def loss_function(self):
        return self._loss_function

    @loss_function.setter
    def loss_function(self, loss_type: str):
        """
        Get the reconstruction loss function. Reconstruction loss function receives the reconstructed and input data.
        :param loss_type:
        :return:
        """
        if loss_type == 'gaussian':
            print("Using Gaussian reconstruction loss")
            if "dec_std" in self.kwargs:
                dec_std = torch.tensor(self.kwargs["dec_std"])
            else:
                dec_std = torch.tensor(1.0)

            def reconstruction_loss(input_data, target):
                loss = torch.square((input_data - target)) / (2 * dec_std ** 2) + torch.log(
                    dec_std) + 0.5 * torch.log(
                    torch.tensor(2 * np.pi))
                # Sum over data dimensions
                loss = torch.sum(loss, dim=tuple(range(1, len(loss.shape))))
                return loss
        elif loss_type == 'bernoulli':
            print("Using Bernoulli reconstruction loss")

            def reconstruction_loss(input_data, target):
                loss = torch.nn.functional.binary_cross_entropy_with_logits(input_data, target, reduction="none")
                # Sum over data dimensions
                loss = torch.sum(loss, dim=tuple(range(1, len(loss.shape))))

                return loss
        elif loss_type == 'bernoulli_avg':
            print("Using Bernoulli reconstruction loss")

            def reconstruction_loss(input_data, target):
                loss = torch.nn.functional.binary_cross_entropy_with_logits(input_data, target, reduction="none")
                # Sum over data dimensions
                loss = torch.mean(loss,dim=tuple(range(1, len(loss.shape))) )

                return loss
        else:
            raise ValueError(f"Reconstruction loss {loss_type} not implemented")
        self._loss_function = reconstruction_loss

    def __call__(self, input_data, target):
        return self.loss_function(input_data, target)


def estimate_entropy(p: torch.distributions.Distribution, n_samples: int = 1000):
    """
    Estimates the entropy of a distribution.
    :param p: distribution
    :param n_samples: number of samples to estimate the entropy
    :return:
    """
    samples = p.sample((n_samples,))
    return -p.log_prob(samples).mean()
