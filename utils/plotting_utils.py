import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from typing import Optional

AVAILABLE_TAB_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink",
                        "tab:gray", "tab:olive", "tab:cyan"]


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
        print(f'Not enough extra dims {extra_dims}, no scatter plot')
        fig = None
        ax = None
    return fig, ax


def add_image_to_ax(data, ax, title=None):
    if len(data.shape) == 1:
        ax.plot(range(len(data)), data)
    else:
        ax.set_aspect('equal', adjustable='box')
        ax.imshow(data, interpolation='nearest', extent=(0, 1, 0, 1))
    if title is not None:
        ax.set_title(title)
    return ax


def plot_images_multi_reconstructions(images, reconstructions):
    """
    Plot images and reconstructions into a plot with 2 rows one for images and one for reconstructions
    :param images: images to plot with shape (batch_size, height, width, channels)
    :param reconstructions: reconstructions to plot with shape (batch_size, height, width, channels)
    :return:
    """
    assert len(images) == len(reconstructions), "Images and reconstructions must have the same number of elements"
    fig, axes = plt.subplots(len(images), reconstructions.shape[1] + 1,
                             figsize=(2.5 * len(images), 2.5 * (reconstructions.shape[1] + 1)))
    for i in range(len(images)):
        add_image_to_ax(images[i], axes[i, 0], title=None)
        for j in range(reconstructions.shape[1]):
            add_image_to_ax(reconstructions[i, j], axes[i, j + 1], title=None)
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            if i == 0:
                if j != 0:
                    # Plot the number of the reconstruction
                    axes[i, j].set_title(f"{j}")
                else:
                    axes[i, j].set_title(f"Original")
        axes[i, j + 1].set_xticks([])
        axes[i, j + 1].set_yticks([])
        axes[0, j + 1].set_title(f"{j + 1}")
    return fig, axes


def plot_images_reconstructions(images, reconstructions):
    """
    Plot images and reconstructions into a plot with 2 rows one for images and one for reconstructions
    :param images: images to plot with shape (batch_size, height, width, channels)
    :param reconstructions: reconstructions to plot with shape (batch_size, height, width, channels)
    :return:
    """
    assert len(images) == len(reconstructions), "Images and reconstructions must have the same number of elements"
    fig, axes = plt.subplots(2, len(images), figsize=(2.5 * len(images), 5))
    for i in range(len(images)):
        add_image_to_ax(images[i], axes[0, i], title='original')
        add_image_to_ax(reconstructions[i], axes[1, i], title='reconstruction')
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])
    return fig, axes


def add_distribution_to_ax(mean, std, ax, n: int, title=None, color=None, dist_text=None):
    if color is None:
        colors = AVAILABLE_TAB_COLORS
    else:
        colors = [color] * n
    if title is not None:
        ax.set_title(title)
    for j in range(n):
        ellipse_j = Ellipse(xy=(mean[j, 0], mean[j, 1]), width=std[j, 0], height=std[j, 1], color=colors[j],
                            linewidth=15, alpha=0.8)
        if dist_text is not None:
            ax.text(mean[j, 0], mean[j, 1], dist_text, color="k", fontsize=12)
        ax.add_artist(ellipse_j)
    return ax


def add_distribution_to_ax_torus(mean, std, ax, n: int, title=None, color=None, dist_text=None):
    if color is None:
        colors = AVAILABLE_TAB_COLORS
    else:
        colors = [color] * n
    ax.set_aspect('equal', adjustable='box')
    if title is not None:
        ax.set_title(title)
    for j in range(n):
        angles0 = np.arctan2(mean[j, 1], mean[j, 0])
        angles1 = np.arctan2(mean[j, 3], mean[j, 2])
        angle_width = np.mean(std[j, 0:2])
        angle_height = np.mean(std[j, 2:])
        ellipse_j = Ellipse(xy=(angles0, angles1), width=angle_width, height=angle_height, color=colors[j],
                            linewidth=15, alpha=0.8)
        if dist_text is not None:
            ax.text(mean[j, 0], mean[j, 1], dist_text, color="k", fontsize=12)
        ax.add_artist(ellipse_j)
    return ax


def add_scatter_to_ax(mean, ax, color=None):
    if color is None:
        colors = AVAILABLE_TAB_COLORS
    else:
        colors = [color] * len(mean)
    for j in range(len(mean)):
        ax.scatter(mean[j, 0], mean[j, 1], marker="*", s=120, color=colors[j])
    return ax


def plot_images_distributions(mean, std, mean_next, std_next, image,
                              image_next, expected_mean, n):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    # Plot first image
    add_image_to_ax(image, axes[0, 0], title='first')

    # Plot second image
    add_image_to_ax(image_next, axes[0, 1], title='second')

    # Plot encoded distribution for first image
    if mean.shape[-1] == 4:
        add_distribution_to_ax_torus(mean, std, axes[1, 0], n, title='before rotation')
        axes[1, 0].set_xlim(-np.pi, np.pi)
        axes[1, 0].set_ylim(-np.pi, np.pi)
        axes[1, 0].set_xlabel("Angle of torus 1")
        axes[1, 0].set_xlabel("Angle of torus 2")
    else:
        add_distribution_to_ax(mean, std, axes[1, 0], n, title='before rotation')
        add_unit_circle_to_ax(axes[1, 0])
        axes[1, 0].set_xlim(-1.2, 1.2)
        axes[1, 0].set_ylim(-1.2, 1.2)

    # Plot encoded distribution for second image
    if mean.shape[-1] == 4:
        add_distribution_to_ax_torus(mean_next, std_next, axes[1, 1], n, title='after rotation')
        axes[1, 1].set_xlim(-np.pi, np.pi)
        axes[1, 1].set_ylim(-np.pi, np.pi)
        axes[1, 1].set_xlabel("Angle of torus 1")
        axes[1, 1].set_xlabel("Angle of torus 2")
        expected_angles = np.stack([np.arctan2(expected_mean[:, 1], expected_mean[:, 0]),
                                    np.arctan2(expected_mean[:, 3], expected_mean[:, 2])], axis=-1)
        add_scatter_to_ax(expected_angles, axes[1, 1])
    else:
        add_distribution_to_ax(mean_next, std_next, axes[1, 1], n, title='after rotation')
        add_unit_circle_to_ax(axes[1, 1])
        axes[1, 1].set_xlim(-1.2, 1.2)
        axes[1, 1].set_ylim(-1.2, 1.2)
        add_scatter_to_ax(expected_mean, axes[1, 1])

    return fig, axes


def plot_embeddings_eval(mean_values, std_values, n, stabilizers, increasing_radius=False):
    """
    Plot the embeddings of the validation data into the circles to visualize overall behavior.
    :param mean_values: embeddings of the validation data
    :param std_values: scale parameters of the embeddings
    :param n: number of distributions per datapoint
    :param stabilizers: number of stabilizers for each datapoint
    :param increasing_radius: if True, the radius of the circles is increasing with the number of images seen
    :return:
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    cmap = mpl.cm.get_cmap('Reds')
    # Plot encoded distribution for first image
    for num_mean, mean in enumerate(mean_values):
        color_ratio = num_mean % (len(mean_values) / stabilizers[num_mean]) / (len(mean_values) / stabilizers[num_mean])
        if increasing_radius:
            radius = 1.0 + color_ratio
        else:
            radius = 1.0

        add_distribution_to_ax(radius * mean, std_values[num_mean], ax, n, color=cmap(color_ratio),
                               dist_text=str(num_mean))
    ax = add_unit_circle_to_ax(ax)
    if not increasing_radius:
        cax = fig.add_axes([0.27, 0.5, 0.5, 0.05])

        ax.set_aspect('equal', adjustable='box')
        fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap),
                     cax=cax, orientation='horizontal')
    return fig, ax


def add_unit_circle_to_ax(ax):
    circle = Ellipse(xy=(0, 0), width=2, height=2, color="k",
                     linewidth=1, alpha=0.7, fill=False)
    ax.add_artist(circle)
    return ax


def plot_embeddings_eval_torus(mean_values, colors=None):
    """
    Plot the embeddings of the validation data in the unfolded torus with colors corresponding to true angles
    :param mean_values: embeddings of the validation data
    :param colors: colors of the points (should depend on the true angles)
    :return:
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    # Plot encoded distribution for first image
    for num_mean, mean in enumerate(mean_values):
        angles = np.stack([np.arctan2(mean[:, 1], mean[:, 0]), np.arctan2(mean[:, 3], mean[:, 2])], axis=-1)
        add_scatter_to_ax(angles, ax, color=colors[num_mean])
    return fig, ax


def plot_eval_images(eval_images, n_rows):
    assert len(eval_images) % n_rows == 0
    fig, axes = plt.subplots(n_rows, int(len(eval_images) / n_rows), figsize=(10, 10))
    for i in range(len(eval_images)):
        add_image_to_ax(eval_images[i], axes[int(i / n_rows), i % n_rows])
    return fig, axes


def save_embeddings_on_circle(mean, std, stabilizers, save_folder: str, dataset_name: str = "",
                              increasing_radius=False):
    n_gaussians = mean.shape[1]  # Assume mean has shape (total_data, num_gaussians, latent)
    if len(stabilizers.shape) == 1:
        for num_unique, unique in enumerate(np.unique(stabilizers)):
            boolean_selection = (stabilizers == unique)
            location = mean[boolean_selection, ...]
            scale = std[boolean_selection, ...]
            if dataset_name.endswith("m"):
                print("Plotting stabilizers equal to 1")
                stabs = np.ones_like(stabilizers[boolean_selection])

            else:
                stabs = stabilizers[boolean_selection]
            fig, axes = plot_embeddings_eval(location, scale, n_gaussians, stabs, increasing_radius)

            axes.set_title(f"Target stabilizers = {unique}")
            if increasing_radius:
                filename = f"radius_{unique}_eval_embeddings.png"
                axes.set_xlim([-2.2, 2.2])
                axes.set_ylim([-2.2, 2.2])
            else:
                filename = f"{unique}_eval_embeddings.png"
                axes.set_xlim([-1.2, 1.2])
                axes.set_ylim([-1.2, 1.2])

            fig.savefig(os.path.join(save_folder, filename), bbox_inches='tight')
    elif len(stabilizers.shape) == 2:
        for num_subgroup in range(stabilizers.shape[-1]):
            for num_unique, unique in enumerate(np.unique(stabilizers[:, num_subgroup])):
                boolean_selection = (stabilizers[:, num_subgroup] == unique)
                location = mean[boolean_selection, :, 2 * num_subgroup: 2 * (num_subgroup + 1)]
                scale = std[boolean_selection, :, 2 * num_subgroup: 2 * (num_subgroup + 1)]

                if dataset_name.endswith("m"):
                    print("Plotting stabilizers equal to 1")
                    stabs = np.ones_like(stabilizers[boolean_selection, num_subgroup])
                else:
                    stabs = stabilizers[boolean_selection, num_subgroup]
                fig, axes = plot_embeddings_eval(location, scale, n_gaussians, stabs, increasing_radius)

                axes.set_title(f"Target stabilizers = {unique}")
                if increasing_radius:
                    filename = f"radius_sg_{num_subgroup}_stabilizer_{unique}_eval_embeddings.png"
                else:
                    filename = f"sg_{num_subgroup}_stabilizer_{unique}_eval_embeddings.png"
                    axes.set_xlim([-1.2, 1.2])
                    axes.set_ylim([-1.2, 1.2])
                fig.savefig(os.path.join(save_folder, filename),
                            bbox_inches='tight')


# Plotting results

def load_plot_val_errors(filepath, ax=None, title=None, error_scale_log: bool = True, fontsize=20):
    """
    Lods the validation errors from a file and plots them
    :param filepath: path to the file
    :param ax: axis to plot on if None a new one is created
    :param title: title of the plot
    :param error_scale_log: if True the error is plotted on a log scale
    :param fontsize: fontsize of the text and ticks
    :return:
    """
    errors = np.load(filepath)
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    else:
        fig = None
    if title is None:
        ax.set_title("Validation error", fontsize=fontsize)
    else:
        ax.set_title(title)
    ax.plot(range(len(errors)), errors)
    if error_scale_log:
        ax.set_yscale('log')
    ax.set_xlabel("Epochs", fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.set_ylabel("Error", fontsize=fontsize)
    ax.grid()
    return fig, ax


def yiq_to_rgb(yiq):
    """
    Convert YIQ colors to RGB.
    :param yiq: yiq colors, shape (n_samples, 3)
    :return:
    """
    conv_matrix = np.array([[1., 0.956, 0.619],
                            [1., -0.272, 0.647],
                            [1., -1.106, 1.703]])
    return np.tensordot(yiq, conv_matrix, axes=((-1,), (-1)))


def yiq_embedding(theta, phi):
    """
    Embed theta and phi into a YIQ color space.
    :param theta: Theta angle in radians
    :param phi: Phi angle in radians
    :return:
    """
    result = np.zeros(theta.shape + (3,))
    steps = 12
    rounding = True
    if rounding:
        theta_rounded = 2 * np.pi * np.round(steps * theta / (2 * np.pi)) / steps
        phi_rounded = 2 * np.pi * np.round(steps * phi / (2 * np.pi)) / steps
        theta = theta_rounded
        phi = phi_rounded
    result[..., 0] = 0.5 + 0.14 * np.cos((theta + phi) * steps / 2) - 0.2 * np.sin(phi)
    result[..., 1] = 0.25 * np.cos(phi)
    result[..., 2] = 0.25 * np.sin(phi)
    return yiq_to_rgb(result)
