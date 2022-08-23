import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from typing import Optional

save_folder = "./visualizations"
os.makedirs(save_folder, exist_ok=True)


def plot_extra_dims(extra_dims, color_labels: Optional = None):
    """
    Plot extra dimensions associated to the invariant part of the latent space representation
    :param extra_dims:
    :param color_labels:
    :return:
    """
    if extra_dims.shape[-1] > 1:  # If extra_dim = 1 scatter cannot be plotted
        fig, ax = plt.subplots(1, 1)
        ax.set_title('Invariant embeddings')
        if color_labels is None:
            plt.scatter(extra_dims[:, 0], extra_dims[:, 1])
        else:
            plt.scatter(extra_dims[:, 0], extra_dims[:, 1], c=color_labels)
    else:
        print('Not enough extra dims, no scatter plot')
        fig = None
        ax = None
    return fig, ax


def add_image_to_ax(image, ax, title=None):
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    if title is not None:
        ax.set_title(title)
    ax.imshow(image, interpolation='nearest', extent=(0, 1, 0, 1))
    return ax


def add_distribution_to_ax(mean, std, ax, n: int, title=None, color="black"):
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    if title is not None:
        ax.set_title(title)
    for j in range(n):
        ellipse_j = Ellipse(xy=(mean[j, 0], mean[j, 1]), width=std[j, 0], height=std[j, 1], color=color,
                            linewidth=15, alpha=0.8)
        ax.add_artist(ellipse_j)
    return ax


def plot_images_distributions(mean, std, mean_next, std_next, image,
                              image_next, n):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    # Plot first image
    add_image_to_ax(image, axes[0, 0], title='first')

    # Plot second image
    add_image_to_ax(image_next, axes[0, 1], title='second')

    # Plot encoded distribution for first image
    add_distribution_to_ax(mean, std, axes[1, 0], n, title='before rotation')

    # Plot encoded distribution for second image
    add_distribution_to_ax(mean_next, std_next, axes[1, 1], n, title='after rotation')

    return fig, axes


def plot_embeddings_eval(mean_values, std_values, n, stabilizers):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    cmap = mpl.cm.get_cmap('Reds')
    # Plot encoded distribution for first image
    for num_mean, mean in enumerate(mean_values):
        color_ratio = num_mean % (len(mean_values) / stabilizers[num_mean]) / (len(mean_values) / stabilizers[num_mean])
        add_distribution_to_ax(mean, std_values[num_mean], ax, n, title='before rotation', color=cmap(color_ratio))
    cax = fig.add_axes([0.27, 0.5, 0.5, 0.05])

    fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap),
                 cax=cax, orientation='horizontal')
    return fig, ax
