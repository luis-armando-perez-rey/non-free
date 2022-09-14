import numpy as np
from sklearn.decomposition import PCA
from typing import Tuple


def get_regular_angles(num_angles: int):
    """
    Creates an array of regular angles in the interval [0, 2pi)
    :param num_angles: number of angles used
    :return:
    """
    return 2 * np.pi * np.arange(0, num_angles) / num_angles


def make_rotation_matrix_2d(angles):
    """
    Create rotation matrices of shape (num_angles, 2, 2)
    Args:
        angles: array of angles shaped as (num_angles,1)

    Returns:

    """
    cos_angle = np.cos(angles)
    sin_angle = np.sin(angles)
    # Stack the cosine and sine of the angles
    matrix = np.stack((cos_angle, -sin_angle,
                       sin_angle, cos_angle),
                      axis=-1)
    # Sha
    output_shape = angles.shape + (2, 2)
    return matrix.reshape(output_shape)


def flatten_array(array) -> np.array:
    """
    FLattens an array of shape (dim1, dim2, dim3, ... dimN) to have shape (dim1*dim2*dim3...*dimN-1, dimN)
    :param array: Input array to be flattened
    :return:
    """
    return array.reshape((-1, array.shape[-1]))


def calculate_zg_flat(z: np.array, n_subgroup: int):
    """
    Calculate the embedding zg corresponding to the n_subgroup  by averaging across the corresponding
    dimension of z (|subgroup1|, |n_subgroup2|, ..., |n_subgroupn_subgroups|, z_dim)
    :param z:
    :param n_subgroup: number of the subgroup transformation
    :return:
    """
    # Average the latent representation across the corre
    mean_latent = np.mean(z, axis=n_subgroup, keepdims=True)
    g_element = z - mean_latent
    return flatten_array(g_element)


def calculate_pca_zg(g_elements_flat):
    """
    Project to the axis of highest variation for the corresponding subgroup
    :param g_elements_flat:
    :return:
    """
    pca = PCA(n_components=2, svd_solver="full")
    pca.fit(g_elements_flat)
    singular_values = pca.singular_values_
    eigenvalues = singular_values ** 2 / (g_elements_flat.shape[0] - 1)
    std = np.sqrt(eigenvalues)
    projected_zg = pca.transform(g_elements_flat)
    projected_zg /= std * np.sqrt(2)
    return projected_zg


def calculate_projected_zg0(projected_zg, k, angles_flat):
    """
    Estimate the embedding for the basepoint h(x_0) = g_k^-1(PCA(zg)) = h(g^-1(x)). The spread of the
    estimates measures the D_LSBD metric for a given k.
    :param projected_zg:
    :param k: Parameter of the inverse transformation
    :param angles_flat: Angles from which each datapoint is generated
    :return:
    """
    inv_rotations = make_rotation_matrix_2d(-k * angles_flat)
    p_zg0 = np.expand_dims(projected_zg, axis=-1)
    p_zg0 = np.matmul(inv_rotations, p_zg0)
    p_zg0 = np.squeeze(p_zg0, axis=-1)
    return p_zg0


def dlsbd_k_torus(z, k):
    """
    D_LSBD metric measured over data with toroidal structure assumes the input z has shape (*num_angles, z_dim)
    :param z: embeddings of the data with shape (*num_angles, z_dim)
    :param k: parameters corresponding to the parametrization of the group representation
    :return:
    """

    # Regular spacing of angles in [0,2pi)
    # assume z_dim has shape (*n_angles, z_dim)
    num_angles_tuple = z.shape[:-1]
    n_subgroups = len(num_angles_tuple)
    assert n_subgroups == len(k), "Parameter k should be the same size as the number of subgroups"
    angles_regular = [get_regular_angles(num_angles) for num_angles in num_angles_tuple]
    # All possible combinations of the regular angles
    angles_combinations = np.array(np.meshgrid(*angles_regular, indexing="ij"))

    # color map
    angles_flat = angles_combinations.reshape(n_subgroups, -1)

    # The mean latent for each group (angles_per_group, z_dim, n_groups)
    equivariances_list = []
    for num_transformation in range(n_subgroups):
        g_element_flat = calculate_zg_flat(z, num_transformation)
        projected = calculate_pca_zg(g_element_flat)
        pc_g = calculate_projected_zg0(projected, k[num_transformation], angles_flat[num_transformation])
        # compute metric
        mean = np.mean(pc_g, axis=0)
        var = np.mean(np.sum((pc_g - mean) ** 2, axis=-1), axis=0)
        equivariances_list.append(var)
    return np.mean(equivariances_list), equivariances_list


def dlsbd_k_cylinder(z, k):
    """
    D_LSBD metric assumes the input z has shape (*num_angles, z_dim)
    """
    # Regular spacing of angles in [0,2pi)
    # assume z_dim has shape (*n_angles, z_dim)
    num_objects = z.shape[0]
    num_angles_tuple = z.shape[1:-1]
    n_subgroups = len(num_angles_tuple)
    print("Number of objects {}, number of subgroups {}".format(num_objects, n_subgroups))
    # assert n_subgroups == len(k), "Parameter k should be the same size as the number of subgroups"
    angles_regular = [get_regular_angles(num_angles) for num_angles in num_angles_tuple]

    # All possible combinations of the regular angles
    angles_combinations = np.array(np.meshgrid(*angles_regular, indexing="ij"))

    # color map
    angles_flat = angles_combinations.reshape(n_subgroups, -1)

    # The mean latent for each group (angles_per_group, z_dim, n_groups)
    equivariances_list = []
    for num_object in range(num_objects):
        for num_transformation in range(n_subgroups):
            g_element_flat = calculate_zg_flat(z[num_object], num_transformation)
            projected = calculate_pca_zg(g_element_flat)
            pc_g = calculate_projected_zg0(projected, k[num_transformation], angles_flat[num_transformation])
            # compute metric
            mean = np.mean(pc_g, axis=0)
            var = np.mean(np.sum((pc_g - mean) ** 2, axis=-1), axis=0)
            equivariances_list.append(var)
    return np.mean(equivariances_list), equivariances_list


def dlsbd(z_loc, k_values, be_verbose: bool = False, factor_manifold: str = "torus") -> Tuple[float, int]:
    available_manifolds = ["torus", "cylinder"]
    assert factor_manifold in available_manifolds, f"Factor manifold {factor_manifold} not available possible values " \
                                                   f"{available_manifolds}"
    metric_values = np.zeros(len(k_values))
    for num_k, k in enumerate(k_values):
        if factor_manifold == "torus":
            metric_values[num_k] = np.sum(dlsbd_k_torus(z_loc, k=k)[0])
        elif factor_manifold == "cylinder":
            metric_values[num_k] = np.sum(dlsbd_k_cylinder(z_loc, k=k)[0])
        if be_verbose:
            print("Combination number {} for k = {}, score = {}".format(num_k, k, metric_values[num_k]))
    score = np.amin(metric_values)
    k_min = k_values[np.argmin(metric_values)]
    return score, k_min


def create_combinations_omega_values_range(start_value: int = -10, end_value: int = 10,
                                           n_transforms: int = 2) -> np.array:
    """
    Creates an array of all possible combinations of the k parameter in the range [start_value, end_value] over
    n_transforms
    :param n_transforms: number of subgroups in the group representation
    :param start_value: start value of the range
    :param end_value: end value of the range
    :return:
    """
    values = range(start_value, end_value + 1)
    num_k = len(values)
    values_repeated = [values] * n_transforms
    k_values = np.array(np.meshgrid(*values_repeated, indexing="ij"))
    k_values = np.moveaxis(k_values.reshape((n_transforms, num_k ** n_transforms)), -1, 0)
    return k_values


def repeat_angles_n_gaussians(angles, n_gaussians):
    """
    Adds a new dimension to the angles array and repeats the values n_gaussians times
    :param angles: array of angles with shape (n_objects, n_angles)
    :param n_gaussians:
    :return:
    """
    return np.repeat(np.expand_dims(angles, axis=-1), n_gaussians, axis=-1)


def apply_inverse_rotation(z, angles):
    """
    Applies the inverse rotation to the latent z
    :param z: latent with shape (n_objects, n_angles, num_gaussians, z_dim)
    :param angles: angles with shape (n_objects, n_angles)
    :return:
    """
    inv_rotations = make_rotation_matrix_2d(-angles)
    inv_z = np.expand_dims(z, axis=-1)
    inv_z = np.matmul(inv_rotations, inv_z)
    inv_z = np.squeeze(inv_z, axis=-1)
    return inv_z


def calculate_dispersion(z_inv):
    """
    Calculates the dispersion of the latent z_inv
    :param z_inv: latent with shape (n_objects, n_angles, num_gaussians, z_dim)
    :return:
    """
    z_inv_mean = np.mean(z_inv, axis=1, keepdims=True)
    dispersion = np.linalg.norm(z_inv - z_inv_mean, axis=-1)
    return dispersion

def dlsbd_metric_mixture(z, angles, average: bool = True):
    """
    Calculates the lsbd metric for embeddings z with shape (num_objects, num_angles, num_gaussians, latent_dim)
    corresponding to a mixture of a certain distribution. Latent dim should be 2 in this case.
    Shape of angles is assumed to be (num_objects, num_angles)
    :param z: embeddings in Z_G
    :param angles: angles used to generate the dataset where embeddings are extracted from
    :return:
    """
    num_gaussians = z.shape[-2]
    angles = repeat_angles_n_gaussians(angles, num_gaussians)
    z_inv = apply_inverse_rotation(z, angles)
    dispersion = calculate_dispersion(z_inv)
    if average:
        dispersion = np.mean(dispersion)
    return dispersion
