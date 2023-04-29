import torch
import os
import numpy as np
from typing import Optional
from sklearn.cluster import DBSCAN


def reload_model(model_dir: str, autoencoder: str = 'None', device: Optional[str] = None):
    """
    Reload the model from the model directory
    :param model_dir: model directory
    :param autoencoder: type of autoencoder
    :param device: device to use and place the model on
    :return:
    """
    if device is None:
        device = 'cpu'
    model_file = os.path.join(model_dir, 'model.pt')
    print("Loading model from: ", model_file)
    model = torch.load(model_file).to(device)
    if autoencoder != 'None':
        decoder_file = os.path.join(model_dir, 'decoder.pt')
        decoder = torch.load(decoder_file).to(device)
        decoder.eval()
    else:
        decoder = None
    model.eval()
    return model, decoder


def get_embeddings(eval_dataloader, model, variablescale=False, device: Optional[str] = None):
    """
    Get the embeddings of the evaluation dataset
    :param eval_dataloader:
    :param model: model used to generate the embeddings
    :param variablescale: whether the scale is variable or fixed
    :return:
    """
    # region GET EMBEDDINGS
    mean_eval = []
    logvar_eval = []
    extra_eval = []
    for num_batch, batch in enumerate(eval_dataloader):
        if device is None:
            mean_eval_, logvar_eval_, extra_eval_ = model(batch[0])
        else:
            mean_eval_, logvar_eval_, extra_eval_ = model(batch[0].to(device))
        mean_eval.append(mean_eval_.detach())
        logvar_eval.append(logvar_eval_.detach())
        extra_eval.append(extra_eval_.detach())
    mean_eval = torch.cat(mean_eval, dim=0)
    logvar_eval = torch.cat(logvar_eval, dim=0)
    extra_eval = torch.cat(extra_eval, dim=0)

    if not (variablescale):
        logvar_eval = -4.6 * torch.ones(logvar_eval.shape).to(logvar_eval.device)
    std_eval = torch.exp(logvar_eval / 2.) / 10
    print("Embeddings shape: ", mean_eval.shape, logvar_eval.shape, std_eval.shape, extra_eval.shape)
    return mean_eval, logvar_eval, std_eval, extra_eval


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_reconstructions(unique_mean: torch.Tensor, unique_extra: torch.Tensor, decoder, extra_dim: int,
                        autoencoder: str):
    if extra_dim == 0:
        x_rec = decoder(unique_mean[:, 0])
    else:
        if autoencoder == "ae_single":
            x_rec = decoder(torch.cat([unique_mean[:, 0], unique_extra], dim=-1))
        elif autoencoder == "vae":
            extra_loc = unique_extra[:, -2 * extra_dim:extra_dim]
            x_rec = decoder(torch.cat([unique_mean[:, 0], extra_loc], dim=-1))
        else:
            x_rec = decoder(
                torch.cat([unique_mean.view((unique_mean.shape[0], -1)), unique_extra], dim=-1))
    return x_rec


def get_n_clusters_noise(embeddings, min_samples=1, eps=0.3):
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(embeddings)
    labels = db.labels_
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    return n_clusters_, n_noise_
