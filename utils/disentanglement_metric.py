import numpy as np
from sklearn.decomposition import PCA
from typing import Tuple
from sklearn.cluster import KMeans


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
    return np.repeat(np.expand_dims(angles, axis=2), n_gaussians, axis=2)


def apply_inverse_rotation(z, angles):
    """
    Applies the inverse rotation to the latent z
    :param z: latent with shape (n_objects, n_angles, num_gaussians, z_dim)
    :param angles: angles with shape (n_objects, n_angles)
    :return:
    """
    # Inverse rotation for torus manifold
    if z.shape[-1] == 4:
        print("Applying inverse rotation for torus manifold")
        inv_z_subspaces = []
        for num_subspace in range(2):
            subspace_angles = angles[..., num_subspace]

            z_subspace = z[..., num_subspace * 2:(num_subspace + 1) * 2]
            print("Z subspace and subspace angles ", z_subspace.shape, subspace_angles.shape, angles.shape)
            inv_z_subspaces.append(apply_inverse_2d_rotation(z_subspace, subspace_angles))
        inv_z = np.concatenate(inv_z_subspaces, axis=-1)
    # Inverse rotation for cylinder manifold
    else:
        print("Applying inverse rotation for cylinder manifold")
        inv_z = apply_inverse_2d_rotation(z, angles)
    return inv_z


def apply_inverse_2d_rotation(z, angles):
    inv_rotations = make_rotation_matrix_2d(-angles)
    inv_z = np.expand_dims(z, axis=-1)
    inv_z = np.matmul(inv_rotations, inv_z)
    inv_z = np.squeeze(inv_z, axis=-1)
    return inv_z


def estimate_mean_inv(z_inv):
    latent_dim = z_inv.shape[-1]
    if latent_dim == 4:
        z_inv_mean = []
        for num_subspace in range(latent_dim // 2):
            z_inv_subspace = z_inv[..., num_subspace * 2:(num_subspace + 1) * 2]
            z_inv_mean.append(np.mean(z_inv_subspace, axis=-3, keepdims=True))
        z_inv_mean = np.concatenate(z_inv_mean, axis=-1)
    else:
        z_inv_mean = np.mean(z_inv, axis=-3, keepdims=True)
    return z_inv_mean


def estimate_kmeans_inv(z_inv, stabilizers):
    latent_dim = z_inv.shape[-1]
    n_gaussians = z_inv.shape[-2]
    if latent_dim == 4:
        z_inv_mean = []
        for num_subspace in range(2):
            obj_stabilizers = np.amin([stabilizers[num_subspace], n_gaussians])
            print("Number of object stabilizers ", obj_stabilizers)
            z_inv_subspace = z_inv[..., num_subspace * 2:(num_subspace + 1) * 2]
            kmeans = KMeans(obj_stabilizers).fit(z_inv_subspace.reshape((-1, z_inv_subspace.shape[-1])))
            z_inv_mean.append(kmeans.cluster_centers_ / np.linalg.norm(kmeans.cluster_centers_, axis=-1, keepdims=True))

    elif latent_dim == 2:
        z_inv_mean = []
        obj_stabilizers = np.amin([stabilizers, n_gaussians])
        print("Number of object stabilizers ", obj_stabilizers)
        kmeans = KMeans(obj_stabilizers).fit(z_inv.reshape((-1, z_inv.shape[-1])))
        z_inv_mean.append(kmeans.cluster_centers_ / np.linalg.norm(kmeans.cluster_centers_, axis=-1, keepdims=True))
    else:
        raise ValueError("Latent dimension not supported")
    return z_inv_mean


def calculate_dispersion(z_inv, z_inv_mean, distance_function: str = "euclidean"):
    """
    Calculates the dispersion of the latent z_inv
    :param z_inv: latent with shape (n_objects, n_angles, num_gaussians, z_dim)
    :return:
    """
    if distance_function == "euclidean":
        dispersion = np.linalg.norm(z_inv - z_inv_mean, axis=-1)
    elif distance_function == "cosine":
        dispersion = 1 - np.sum(z_inv * z_inv_mean, axis=-1) / (
                np.linalg.norm(z_inv, axis=-1) * np.linalg.norm(z_inv_mean, axis=-1))
    elif distance_function == "cross-entropy":
        raise NotImplementedError
    elif distance_function == "chamfer":
        dispersion = matrix_dist_numpy(z_inv, z_inv_mean).min(-1)
        print("Dispersion shape!!", dispersion.shape)
        dispersion = np.mean(dispersion)
    else:
        raise NotImplementedError
    return dispersion


def matrix_dist_numpy(z_mean_next, z_mean_pred):
    latent_dim = z_mean_next.shape[-1]
    if latent_dim != 3:
        print("Latent dimension is not 3", (np.expand_dims(z_mean_pred, 1) - np.expand_dims(z_mean_next, 2)).shape)
        return ((np.expand_dims(z_mean_pred, 1) - np.expand_dims(z_mean_next, 2)) ** 2).sum(-1)

    else:
        return ((np.expand_dims(z_mean_pred, 1) - np.expand_dims(z_mean_next, 2)) ** 2).sum(-1).sum(-1)


def dlsbd_metric_mixture(z, angles, stabilizers, average: bool = True, distance_function: str = "euclidean"):
    """
    Calculates the lsbd metric for embeddings z with shape (num_objects, num_angles, num_gaussians, latent_dim)
    corresponding to a mixture of a certain distribution. Latent dim should be 2 in this case.
    Shape of angles is assumed to be (num_objects, num_angles)
    :param z: embeddings in Z_G
    :param angles: angles used to generate the dataset where embeddings are extracted from
    :param stabilizers: number of stabilizers for each object and subspace. Has shape (num_objects, num_subspaces)
    :return:
    """
    num_gaussians = z.shape[-2]
    num_subspaces = z.shape[-1] // 2
    angles = repeat_angles_n_gaussians(angles, num_gaussians)
    z_inv = apply_inverse_rotation(z, angles)
    dispersion_values = []
    for num_object in range(z_inv.shape[0]):
        z_inv_object = z_inv[num_object]
        # z_inv_mean = estimate_mean_inv(z_inv_object)
        z_inv_mean = estimate_kmeans_inv(z_inv_object, stabilizers[num_object])
        # print(f"Stabilizers of num object {num_object}", stabilizers[num_object])
        for num_subspace in range(num_subspaces):
            # print("Number of subspace", num_subspace)
            z_inv_object_subspace = z_inv_object[..., num_subspace * 2:(num_subspace + 1) * 2]
            z_inv_mean_subspace = np.expand_dims(z_inv_mean[num_subspace], axis=0)
            # print("ZINV MEAN SUBSPACE", z_inv_mean_subspace.shape, z_inv_object_subspace.shape)
            dispersion_obj = calculate_dispersion(z_inv_object_subspace,
                                                  z_inv_mean_subspace,
                                                  distance_function=distance_function)
            print(
                f"Dispersion for object {num_object}, with num stabilizers {stabilizers[num_object]} num subspace {num_subspace} = {dispersion_obj} ")
            # Get dispersion values per object
            dispersion_values.append(np.mean(dispersion_obj))

    if average:
        dispersion = np.mean(dispersion_values)
    else:
        dispersion = np.array(dispersion_values)

    return dispersion
