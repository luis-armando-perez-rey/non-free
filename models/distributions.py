import torch
import torch.distributions as D


class MixtureDistribution(D.MixtureSameFamily):
    @property
    def mode(self):
        pass

    def __init__(self, input_mean, input_logvar, encoder_distribution_type: str):
        device = input_mean.device
        self.encoder_distribution_type = encoder_distribution_type
        self.input_mean = input_mean
        self.input_logvar = input_logvar
        self.components = encoder_distribution_type
        mix = D.Categorical(torch.ones((self.input_mean.shape[0], self.input_mean.shape[1])).to(device))
        super().__init__(mix, self.components)

    @property
    def components(self):
        return self._components

    @components.setter
    def components(self, encoder_distribution_type: str):
        if encoder_distribution_type == "gaussian-mixture":
            components = D.Independent(D.Normal(self.input_mean, torch.exp(self.input_logvar)), 1)
        elif encoder_distribution_type == "von-mises-mixture":
            angle = torch.atan2(self.input_mean[..., -2], self.input_mean[..., -1])
            components = D.von_mises.VonMises(loc=angle, concentration=1 / torch.exp(self.input_logvar[..., -1]))
        else:
            components = None
            ValueError(f"{encoder_distribution_type} not available")
        self._components = components

    def entropy(self):
        NotImplementedError()

    def enumerate_support(self, expand=True):
        NotImplementedError()

    def icdf(self, value):
        NotImplementedError()

    def rsample(self, sample_shape=torch.Size()):
        NotImplementedError()

    @property
    def sample_latent(self):
        if self.encoder_distribution_type == "von-mises-mixture":
            def sample_latent(tuple_samples):
                angle = self.sample(tuple_samples)
                return torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)
        else:
            sample_latent = self.sample
        return sample_latent
