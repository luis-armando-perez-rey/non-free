import torch
import torch.distributions as D
import numpy as np
from typing import Optional


class MixturePrior(D.Distribution):
    def __init__(self, batch_size: int, num_components: int, prior_type: str, device, **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.num_components = num_components
        self.device = device
        self.num_components = num_components
        self.prior_type = prior_type
        self.kwargs = kwargs

    @property
    def prior(self):
        return self._prior

    @prior.setter
    def prior(self, prior_type: str):
        if prior_type == "von-mises-mixture":
            if "concentation" in self.kwargs:
                concentration = self.kwargs["concentration"]
            else:
                concentration = 0.01
                self._prior = D.von_mises.VonMises(
                    torch.tensor(2 * np.pi) * torch.rand((self.batch_size, self.num_components)).to(self.device),
                    torch.tensor(concentration) * torch.ones((self.batch_size, self.num_components)).to(self.device))


class MixtureDistribution(D.MixtureSameFamily):

    def __init__(self, input_mean, input_logvar, encoder_distribution_type: str):
        device = input_mean.device
        self.encoder_distribution_type = encoder_distribution_type
        self.input_mean = input_mean
        self.input_logvar = input_logvar
        self.components = encoder_distribution_type
        mix = D.Categorical(torch.ones((self.input_mean.shape[0], self.input_mean.shape[1])).to(device))
        if self.components is not None:
            super().__init__(mix, self.components)

    @property
    def components(self) -> Optional[D.Distribution]:
        return self._components

    @components.setter
    def components(self, encoder_distribution_type: str):
        if encoder_distribution_type == "gaussian-mixture":
            components = D.Independent(D.Normal(self.input_mean, torch.exp(self.input_logvar / 2.0)), 1)
        elif encoder_distribution_type == "von-mises-mixture":
            angle = torch.atan2(self.input_mean[..., -2], self.input_mean[..., -1])
            components = D.von_mises.VonMises(loc=angle, concentration=1 / torch.exp(self.input_logvar[..., -1]))
        else:
            components = None
            # ValueError(f"{encoder_distribution_type} not available")
        self._components: Optional[D.Distribution] = components

    @property
    def sample_latent(self):
        if self.encoder_distribution_type == "von-mises-mixture":
            # Project the Von Mises angle samples to the unit circle
            def sample_latent(tuple_samples):
                angle = self.sample(tuple_samples)
                return torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)
        elif self.encoder_distribution_type == "gaussian-mixture":
            # Sample from gaussian mixture and project to the unit circle
            def sample_latent(tuple_samples):
                sample = self.sample(tuple_samples)
                sample = torch.nn.functional.normalize(sample)
                return sample
        else:
            sample_latent = self.sample
        return sample_latent

    def approximate_kl(self, p: D.Distribution, n_samples: int = 20) -> torch.Tensor:
        # Sample from the current distribution
        posterior_samples = self.sample((n_samples,))
        return self.log_prob(posterior_samples).mean() - p.log_prob(posterior_samples).mean()

    def approximate_kl_components(self, p: D.Distribution, n_samples: int = 20) -> torch.Tensor:
        # Sample from the current distribution
        posterior_samples = self.components.sample((n_samples,))
        # Calculate the approximate entropy of the posterior
        component_log_prob = self.components.log_prob(posterior_samples).mean()
        # Calculate the approximate cross entropy of the prior. Assume that the prior is the same for each batch element
        prior_log_prob = p.log_prob(posterior_samples.view((-1, posterior_samples.shape[1]))).mean()
        return component_log_prob - prior_log_prob

    def entropy(self):
        return NotImplementedError()

    def enumerate_support(self, expand=True):
        return NotImplementedError()

    def icdf(self, value):
        return NotImplementedError()

    def rsample(self, sample_shape=torch.Size()):
        return NotImplementedError()

    @property
    def mode(self):
        return NotImplementedError()


def get_z_values(p: MixtureDistribution, extra: torch.tensor, n_samples: int, autoencoder_type: str):
    """
    Sample from posterior over Z_G and join with extra variables from Z_I (orbit encoding).
    :param p: Posterior distribution over Z_G
    :param extra: Encodings on Z_I
    :param n_samples: Number of samples to draw from posterior over Z_G
    :param autoencoder_type: Type of autoencoder to be used
    :return:
    """
    if autoencoder_type.startswith("ae"):
        z = p.input_mean
        if extra is not None:
            extra_repeated = extra.unsqueeze(1).repeat(1, z.shape[1], 1)
            z = torch.cat([z.view((z.shape[0], z.shape[1], -1)), extra_repeated], dim=-1)
    elif autoencoder_type == "vae":
        # z = torch.movedim(p.sample_latent((n_samples,)), 0, 1)
        z = p.input_mean
        if extra is not None:
            # Sample from posterior over Z_I
            loc_extra = extra[..., :extra.shape[-1] // 2]
            logvar_extra = extra[..., extra.shape[-1] // 2:]
            # Uncomment for fixing the scale parameter of the extra values
            logvar_extra = -4.6 * torch.ones(logvar_extra.shape).to(logvar_extra.device)
            p_orbit = torch.distributions.Normal(loc_extra, torch.exp(logvar_extra / 2.0))
            extra_sample = p_orbit.sample()
            # Repeat samples from Z_I to match the number of samples from Z_G
            extra_repeated = extra_sample.unsqueeze(1).repeat(1, z.shape[1], 1)
            z = torch.cat([z.view((z.shape[0], z.shape[1], -1)), extra_repeated], dim=-1)
    elif autoencoder_type == "vae_mixture":
        z = torch.movedim(p.sample_latent((n_samples,)), 0, 1)
        if extra is not None:
            # Sample from posterior over Z_I
            loc_extra = extra[..., :extra.shape[-1] // 2]
            logvar_extra = extra[..., extra.shape[-1] // 2:]
            # Uncomment for fixing the scale parameter of the extra values
            # Allow variable location for the extra values
            # logvar_extra = -4.6 * torch.ones(logvar_extra.shape).to(logvar_extra.device)
            p_orbit = torch.distributions.Normal(loc_extra, torch.exp(logvar_extra / 2.0))
            extra_sample = p_orbit.sample()
            z = torch.reshape(z, (z.shape[0], -1))
            z = torch.cat([z, extra_sample], dim=-1)

    else:
        z = None
        ValueError(f"Autoencoder type {autoencoder_type} not defined")
    # Do not change order! Append of extra dimension should be done after Von-Mises projection
    return z


def get_prior(batch_size: int, num_components: int, latent_dim: int, prior_type: str, device, **kwargs):
    """
    Returns a prior distribution with location parameter with shape (batch_size, num_components, latent_dim)
    :param batch_size: Batch size
    :param num_components: Number of components in the mixture
    :param latent_dim: Dimension of the latent space
    :param prior_type: Type of prior distribution
    :param device: Device to run the model on
    :param kwargs: Additional arguments for the prior distribution. E.g. concentration for Von Mises mixture and
    Gaussian mixture (inverse scale).
    :return:
    """
    if prior_type == "von-mises-mixture":
        # print("Using von-mises mixture prior")
        if "concentation" in kwargs:
            concentration = kwargs["concentration"]
        else:
            concentration = 0.01
        mix = D.Categorical(torch.ones((batch_size, num_components)).to(device))
        angle = torch.tensor(2 * np.pi) * torch.rand((batch_size, num_components)).to(device)
        components = torch.distributions.von_mises.VonMises(
            angle, torch.tensor(concentration).to(device))
        prior = D.MixtureSameFamily(mix, components)
    elif prior_type == "gaussian-mixture":
        # print("Using gaussian mixture prior")
        # print("NOTE: Gaussian mixture prior only implemented for embeddings on the circle")
        if "concentation" in kwargs:
            concentration = kwargs["concentration"]
        else:
            concentration = 1.0
        angles = torch.tensor(2 * np.pi) * torch.rand((batch_size, num_components)).to(device)
        mean = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        mix = D.Categorical(torch.ones((batch_size, num_components)).to(device))
        component = D.Independent(D.Normal(mean, scale=1.0 / concentration), 1)
        prior = D.MixtureSameFamily(mix, component)
    elif prior_type == "gaussian":
        # print("Using gaussian prior")
        prior = D.Independent(
            D.Normal(torch.zeros((batch_size, latent_dim)).to(device), torch.ones((batch_size, latent_dim)).to(device)),
            1)
    else:
        prior = None
        ValueError(f"{prior_type} not available")
    return prior
