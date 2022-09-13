import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

"""
Code obtained from the colab in https://implicit-pdf.github.io/
Please cite the following work if using this code:
@inproceedings{implicitpdf2021,
  title = {Implicit-PDF: Non-Parametric Representation of Probability Distributions on the Rotation Manifold},
  author = {Murphy, Kieran A and Esteves, Carlos and Jampani, Varun and Ramalingam, Srikumar and Makadia, Ameesh},
  booktitle = {Proceedings of the 38th International Conference on Machine Learning},
  pages = {7882--7893},
  year = {2021}
}
Re-implemented the tensorflow-graphics version of the function tfg.geometry.transformation.euler.from_rotation_matrix 
to use numpy arrays to avoid using tensorflow-graphics due to some import problems. 
"""


def safe_nonzero_sign(x):
    ones = np.ones_like(x)
    return np.where(np.greater_equal(x, 0), ones, -ones)


def select_eps_for_division(dtype):
    """Selects a small epsilon value for division.
  Args:
    dtype: The `dtype` of the tensor to be divided.
  Returns:
    A small epsilon value of the same `dtype` as `tensor`.
  """
    return 10.0 * np.finfo(dtype).tiny


def from_rotation_matrix(rotation_matrix):
    """Converts rotation matrices to Euler angles.
    The rotation matrices are assumed to have been constructed by rotation around
    the $$x$$, then $$y$$, and finally the $$z$$ axis.
    Note:
    There is an infinite number of solutions to this problem. There are
    Gimbal locks when abs(rotation_matrix(2,0)) == 1, which are not handled.
    Note:
    In the following, A1 to An are optional batch dimensions.
    Args:
    rotation_matrix: A tensor of shape `[A1, ..., An, 3, 3]`, where the last two
      dimensions represent a rotation matrix.
    name: A name for this op that defaults to "euler_from_rotation_matrix".
    Returns:
    A tensor of shape `[A1, ..., An, 3]`, where the last dimension represents
    the three Euler angles.
    Raises:
    ValueError: If the shape of `rotation_matrix` is not supported.
    """

    def general_case(rot_matrix, entry20, epsilon):
        """Handles the general case."""
        theta_y = -np.arcsin(entry20)
        sign_cos_theta_y = safe_nonzero_sign(np.cos(theta_y))
        r00 = rot_matrix[..., 0, 0]
        r10 = rot_matrix[..., 1, 0]
        r21 = rot_matrix[..., 2, 1]
        r22 = rot_matrix[..., 2, 2]
        r00 = safe_nonzero_sign(r00) * epsilon + r00
        r22 = safe_nonzero_sign(r22) * epsilon + r22
        # cos_theta_y evaluates to 0 on Gimbal locks, in which case the output of
        # this function will not be used.
        theta_z = np.arctan2(r10 * sign_cos_theta_y, r00 * sign_cos_theta_y)
        theta_x = np.arctan2(r21 * sign_cos_theta_y, r22 * sign_cos_theta_y)
        angles = np.stack((theta_x, theta_y, theta_z), axis=-1)
        return angles

    def gimbal_lock(rot_matrix, entry20, epsilon):
        """Handles Gimbal locks."""
        r01 = rot_matrix[..., 0, 1]
        r02 = rot_matrix[..., 0, 2]
        sign_r20 = safe_nonzero_sign(entry20)
        r02 = safe_nonzero_sign(r02) * epsilon + r02
        theta_x = tf.atan2(-sign_r20 * r01, -sign_r20 * r02)
        theta_y = -sign_r20 * tf.constant(np.pi / 2.0, dtype=entry20.dtype)
        theta_z = tf.zeros_like(theta_x)
        angles = tf.stack((theta_x, theta_y, theta_z), axis=-1)
        return angles

    r20 = rotation_matrix[..., 2, 0]
    eps_addition = select_eps_for_division(rotation_matrix.dtype)
    general_solution = general_case(rotation_matrix, r20, eps_addition)
    gimbal_solution = gimbal_lock(rotation_matrix, r20, eps_addition)
    is_gimbal = np.equal(np.abs(r20), 1)
    gimbal_mask = np.stack((is_gimbal, is_gimbal, is_gimbal), axis=-1)
    return np.where(gimbal_mask, gimbal_solution, general_solution)


def visualize_so3_probabilities(rotations,
                                probabilities,
                                rotations_gt=None,
                                ax=None,
                                fig=None,
                                display_threshold_probability=0,
                                show_color_wheel=True,
                                canonical_rotation=np.eye(3)):
    """Plot a single distribution on SO(3) using the tilt-colored method.
    Args:
      rotations: [N, 3, 3] numpy array of rotation matrices
      probabilities: [N] numpy array of probabilities
      rotations_gt: [N_gt, 3, 3] or [3, 3] ground truth rotation matrices
      ax: The matplotlib.pyplot.axis object to paint
      fig: The matplotlib.pyplot.figure object to paint
      display_threshold_probability: The probability threshold below which to omit
        the marker
      show_color_wheel: If True, display the explanatory color wheel which matches
        color on the plot with tilt angle
      canonical_rotation: A [3, 3] rotation matrix representing the 'display
        rotation', to change the view of the distribution.  It rotates the
        canonical axes so that the view of SO(3) on the plot is different, which
        can help obtain a more informative view.
    Returns:
      A matplotlib.pyplot.figure object, or a tensor of pixels if to_image=True.
    """

    def _show_single_marker(ax, rotation, marker, edgecolors=True,
                            facecolors=False):

        eulers = from_rotation_matrix(rotation)
        xyz = rotation[:, 0]
        tilt_angle = eulers[0]
        longitude = np.arctan2(xyz[0], -xyz[1])
        latitude = np.arcsin(xyz[2])

        color = cmap(0.5 + tilt_angle / 2 / np.pi)
        ax.scatter(longitude, latitude, s=2500,
                   edgecolors=color if edgecolors else 'none',
                   facecolors=facecolors if facecolors else 'none',
                   marker=marker,
                   linewidth=4)

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='mollweide')
    if rotations_gt is not None and len(rotations_gt.shape) == 2:
        rotations_gt = rotations_gt[np.newaxis]

    display_rotations = rotations @ canonical_rotation
    cmap = plt.cm.hsv
    scatterpoint_scaling = 4e3
    eulers_queries = from_rotation_matrix(display_rotations.numpy())
    xyz = display_rotations[:, :, 0]
    tilt_angles = eulers_queries[:, 0]

    longitudes = np.arctan2(xyz[:, 0], -xyz[:, 1])
    latitudes = np.arcsin(xyz[:, 2])

    which_to_display = (probabilities > display_threshold_probability)

    if rotations_gt is not None:
        # The visualization is more comprehensible if the GT
        # rotation markers are behind the output with white filling the interior.
        display_rotations_gt = rotations_gt @ canonical_rotation

        for rotation in display_rotations_gt:
            _show_single_marker(ax, rotation, 'o')
        # Cover up the centers with white markers
        for rotation in display_rotations_gt:
            _show_single_marker(ax, rotation, 'o', edgecolors=False,
                                facecolors='#ffffff')

    # Display the distribution
    ax.scatter(
        longitudes[which_to_display],
        latitudes[which_to_display],
        s=scatterpoint_scaling * probabilities[which_to_display],
        c=cmap(0.5 + tilt_angles[which_to_display] / 2. / np.pi))

    ax.grid()
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    if show_color_wheel:
        # Add a color wheel showing the tilt angle to color conversion.
        ax = fig.add_axes([0.86, 0.17, 0.12, 0.12], projection='polar')
        theta = np.linspace(-3 * np.pi / 2, np.pi / 2, 200)
        radii = np.linspace(0.4, 0.5, 2)
        _, theta_grid = np.meshgrid(radii, theta)
        colormap_val = 0.5 + theta_grid / np.pi / 2.
        ax.pcolormesh(theta, radii, colormap_val.T, cmap=cmap)
        ax.set_yticklabels([])
        ax.set_xticklabels([r'90$\degree$', None,
                            r'180$\degree$', None,
                            r'270$\degree$', None,
                            r'0$\degree$'], fontsize=14)
        ax.spines['polar'].set_visible(False)
        plt.text(0.5, 0.5, 'Tilt', fontsize=14,
                 horizontalalignment='center',
                 verticalalignment='center', transform=ax.transAxes)
    plt.show()
    return fig
