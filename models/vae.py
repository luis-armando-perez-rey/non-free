import torch
import numpy as np


def get_reconstruction_loss(reconstruction_str: str, **kwargs):
    if reconstruction_str == 'gaussian':
        print("Using Gaussian reconstruction loss")
        if "dec_std" in kwargs:
            dec_std = torch.tensor(kwargs["dec_std"])
        else:
            dec_std = torch.tensor(1.0)

        def reconstruction_loss(input_data, target):
            loss = torch.square((input_data - target)) / (2 * dec_std ** 2) + torch.log(dec_std) + 0.5 * torch.log(
                torch.tensor(2 * np.pi))
            loss = torch.sum(loss, dim=tuple(range(1, len(loss.shape))))
            return loss
    elif reconstruction_str == 'bernoulli':
        print("Using Bernoulli reconstruction loss")

        def reconstruction_loss(input_data, target):
            loss = torch.nn.functional.binary_cross_entropy_with_logits(input_data, target, reduction="none")
            loss = torch.sum(loss, dim=tuple(range(1, len(loss.shape))))
            return loss
    else:
        raise ValueError(f"Reconstruction loss {reconstruction_str} not implemented")
    return reconstruction_loss
