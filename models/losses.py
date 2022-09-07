import torch
import numpy as np


class EquivarianceLoss:
    def __init__(self, loss_type: str, encoder_dist_type: str):
        self.enc_dist_type = encoder_dist_type
        self.location_parameter_function = encoder_dist_type
        self.loss_function = loss_type

    @property
    def location_parameter_function(self):
        return self._location_parameter_function

    @location_parameter_function.setter
    def location_parameter_function(self, encoder_dist_type):
        """
        Get the function that returns the location parameter of the encoding distribution. This is useful when using
        mixture distributions.
        :param encoder_dist_type:
        :return:
        """
        if encoder_dist_type == "gaussian-mixture":
            def location_parameter_function(p: torch.distributions.MixtureSameFamily):
                return p.component_distribution.base_dist.mean
        elif encoder_dist_type == "von-mises-mixture":
            def location_parameter_function(p: torch.distributions.MixtureSameFamily):
                return p.component_distribution.mean
        else:
            location_parameter_function = None
            ValueError(f"{self.enc_dist_type} not available")
        self._location_parameter_function = location_parameter_function

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
        elif loss_type == "chamfer":
            def equivariance_loss_function(p: torch.distributions.MixtureSameFamily,
                                           p_next: torch.distributions.MixtureSameFamily):
                mean = self.location_parameter_function(p)
                mean_next = self.location_parameter_function(p_next)
                loss = ((mean.unsqueeze(1) - mean_next.mean.unsqueeze(2)) ** 2).sum(-1).min(dim=-1)[0].sum(
                    dim=-1).mean()
                return loss
        elif loss_type == "euclidean":
            def equivariance_loss_function(p: torch.distributions.MixtureSameFamily,
                                           p_next: torch.distributions.MixtureSameFamily):
                mean = self.location_parameter_function(p)
                mean_next = self.location_parameter_function(p_next)
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

    @staticmethod
    def cross_entropy_mixture(p1, p2: torch.distributions.Distribution, n_samples: int = 20):
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
        if loss_type == "info-nce":
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
        else:
            raise ValueError(f"Reconstruction loss {loss_type} not implemented")
        self._loss_function = reconstruction_loss