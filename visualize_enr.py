import argparse
import pickle
import numpy as np
import os
import sys
import torch
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from utils.dataset_utils import get_dataset
from utils.nn_utils import get_rotated_mean
from utils.plotting_utils import plot_extra_dims, plot_images_distributions, \
    plot_mixture_neurreps, add_image_to_ax, add_distribution_to_ax_torus, save_embeddings_on_circle
from utils.plotting_utils_so3 import visualize_so3_probabilities
from utils.torch_utils import torch_data_to_numpy
from sklearn.decomposition import PCA
from datasets.equiv_dset import EquivDataset
from utils.nn_utils import hitRate_generic

CWD = os.getcwd()
PROJECT_PATH = os.path.dirname(CWD)
LSBD_PATH = os.path.join(PROJECT_PATH, "lsbd-vae")

sys.path.append(LSBD_PATH)

print("Appended ", PROJECT_PATH)
print("Appended ", LSBD_PATH)
from lsbd_vae.metrics.dlsbd_metric import dlsbd, create_combinations_omega_values_range

# region PARSE ARGUMENTS
parser = argparse.ArgumentParser()
parser.add_argument('--save-folder', type=str, default='checkpoints', help='Path to saved model')
parser.add_argument('--dataset', type=str, default='dataset', help='Dataset')
parser.add_argument('--dataset_name', nargs="+", type=str, default=['4'], help='Dataset name')
args_eval = parser.parse_args()
# endregion

model_dir = os.path.join(".", "saved_models", args_eval.save_folder)
model_file = os.path.join(model_dir, 'model.pt')
decoder_file = os.path.join(model_dir, 'decoder.pt')
meta_file = os.path.join(model_dir, 'metadata.pkl')
args = pickle.load(open(meta_file, 'rb'))['args']
device = torch.device("cuda:" + args.gpu if torch.cuda.is_available() else "cpu")
# torch.manual_seed(42)
torch.cuda.empty_cache()
print(args)
save_folder = os.path.join(".", "visualizations", args.model_name)
os.makedirs(save_folder, exist_ok=True)


def get_pca_model(data, n_components=2):
    pca = PCA(n_components=n_components, svd_solver="full")
    pca.fit(data)
    return pca


def get_normalizing_constant(pca, n_data):
    singular_values = pca.singular_values_
    eigenvalues = singular_values ** 2 / (n_data - 1)
    std_projection = np.sqrt(eigenvalues)
    normalizing_constant = std_projection * np.sqrt(2)
    return normalizing_constant


# region SETUP NEPTUNE
if args.neptune_user != "":
    from utils.neptune_utils import reload_neptune_run

    neptune_id_file = os.path.join(model_dir, 'neptune.txt')
    run = reload_neptune_run(args.neptune_user, "non-free", neptune_id_file)
else:
    run = None
# endregion

# region LOAD DATASET
if args.dataset != "symmetric_solids" and args.dataset != "modelnetso3":
    dset, eval_dset = get_dataset(args.data_dir, args.dataset, args.dataset_name, so3_matrices=True)
else:
    dset, eval_dset = get_dataset(args.data_dir, args.dataset, args.dataset_name, so3_matrices=False)
train_loader = torch.utils.data.DataLoader(dset, batch_size=100, shuffle=False)
# endregion

# region GET MODEL
device = 'cpu'

model = torch.load(model_file).to(device)
model.eval()
# endregion

# region GET IMAGES
img, img_next, action, n_stabilizers = next(iter(train_loader))
action = action.squeeze(1)
img_shape = np.array(img.shape[1:])
# endregion

# region GET NUMPY DATA
# Get the numpy array versions of the images
npimages = torch_data_to_numpy(img)
npimages_next = torch_data_to_numpy(img_next)
npimages_eval = eval_dset.flat_images_numpy
# endregion

# region GET EMBEDDINGS
# Calculate the parameters obtained by the models
mean = model.encode(img.to(device))
mean_next = model.encode(img_next.to(device))

# Obtain the values as numpy arrays
mean_numpy = mean.detach().cpu().numpy()
mean_next = mean_next.detach().cpu().numpy()
mean_rot = model.act(mean, action).detach().cpu().numpy()
action = action.detach().cpu().numpy()
print("MEAN SHAPE!!!", mean_numpy.shape, np.unique(mean_numpy))
# endregion

# region CALCULATE HITRATE
# Setup torch dataset use separate data as validation
if (args.dataset != "symmetric_solids") and (args.dataset != "modelnetso3"):
    dset_val = EquivDataset(f'{args.data_dir}/{args.dataset}/',
                            list_dataset_names=[dataset_name + "_val" for dataset_name in args.dataset_name],
                            max_data_per_dataset=-1, so3_matrices=True)
else:
    dset_val = EquivDataset(f'{args.data_dir}/{args.dataset}/',
                            list_dataset_names=[dataset_name + "_val" for dataset_name in args.dataset_name],
                            max_data_per_dataset=-1)
val_loader = torch.utils.data.DataLoader(dset_val, batch_size=args.batch_size, shuffle=True)
total_batches = len(val_loader)
mu_hitrate = 0
for batch_idx, (image, img_next, action) in enumerate(val_loader):
    batch_size = image.shape[0]
    encoded_image = model.encode(image)
    encoded_image_next = model.encode(img_next)
    action = action.to(device).squeeze(1)
    encoded_image_transformed = model.act(encoded_image, action)
    encoded_image_transformed_flat = encoded_image_transformed.view((batch_size, -1))
    encoded_image_next_flat = encoded_image_next.view((batch_size, -1))
    dist_matrix = ((encoded_image_transformed_flat.unsqueeze(0) - encoded_image_next_flat.unsqueeze(1)) ** 2).sum(-1)
    hitrate = hitRate_generic(dist_matrix, batch_size)
    mu_hitrate += hitrate.item()
mu_hitrate /= total_batches
model_path = os.path.join(args.checkpoints_dir, args.model_name)
np.save(f'{model_path}/errors_hitrate.npy', [mu_hitrate])
print(f"Hitrate: {mu_hitrate}")
if run is not None:
    run["metrics/hitrate"].log(mu_hitrate)
# endregion


# region PLOT ROTATED EMBEDDINGS
flat_mean = mean_numpy.reshape(mean_numpy.shape[0], -1)
flat_mean_next = mean_next.reshape(mean_next.shape[0], -1)
flat_mean_rot = mean_rot.reshape(mean_rot.shape[0], -1)
# Get PCA model projection
n_data = flat_mean.shape[0]
pca_model = get_pca_model(flat_mean)
normalizing_constant = get_normalizing_constant(pca_model, n_data)

projected_mean = np.expand_dims(pca_model.transform(flat_mean) / normalizing_constant, axis=1)
projected_mean_next = np.expand_dims(pca_model.transform(flat_mean_next) / normalizing_constant, axis=1)
projected_mean_rot = np.expand_dims(pca_model.transform(flat_mean_rot) / normalizing_constant, axis=1)

# logvar_eval = -4.6 * np.ones_like(projected_mean)
std_next = np.ones_like(projected_mean) * 0.1
std = std_next
for i in range(5):
    print(f"Plotting example {i}")
    fig, axes = plot_images_distributions(mean=projected_mean[i], std=std[i], mean_next=projected_mean_next[i],
                                          std_next=std_next[i],
                                          image=npimages[i], image_next=npimages_next[i],
                                          expected_mean=projected_mean_rot[i], n=1, set_limits=True)
    if run is not None:
        run[f"image_distribution {i}"].upload(plt.gcf())
    plt.savefig(os.path.join(save_folder, f"image_pair_{i}.png"), bbox_inches='tight')

    plot_mixture_neurreps(mean_numpy[i])
    plt.savefig(os.path.join(save_folder, f"test_mixture_{i}.png"), bbox_inches='tight')
    add_image_to_ax(npimages[i])

    plt.savefig(os.path.join(save_folder, f"test_image_{i}.png"), bbox_inches='tight')

print("Dataset", args.dataset)
orbitnum = 0
if args.dataset != "symmetric_solids" and args.dataset != "modelnetso3":
    print(len(eval_dset.data[orbitnum]))
    eval_mean = model.encode(torch.tensor(eval_dset.flat_images, dtype=img.dtype).to(device))
    projected_eval = np.expand_dims(
        pca_model.transform(eval_mean.reshape(eval_mean.shape[0], -1).detach().cpu().numpy()) / normalizing_constant,
        axis=1)
    std_eval = np.ones_like(projected_eval) * 0.1
    print("Projected eval shape", projected_eval.shape, "Std eval shape", std_eval.shape)
    figures = save_embeddings_on_circle(projected_eval, std_eval, eval_dset.flat_stabs, save_folder,
                                        args.dataset_name[
                                            orbitnum], increasing_radius=True)
    if run is not None:
        print(figures)
        run["embeddings_on_circle"].upload(figures[0])
else:
    eval_indices = np.arange(0, len(eval_dset.flat_images), len(eval_dset.flat_images) // 36)
    eval_mean = model.encode(torch.tensor(eval_dset.flat_images[eval_indices], dtype=img.dtype).to(device))

# endregion


if args.dataset != "symmetric_solids":

    # region GET EMBEDDINGS DLSBD
    flat_images_tensor = torch.Tensor(eval_dset.flat_images).to(device)  # transform to torch tensor
    eval_tensor_dset = torch.utils.data.TensorDataset(flat_images_tensor)  # create your datset
    eval_dataloader = torch.utils.data.DataLoader(eval_tensor_dset, batch_size=args.batch_size)
    mean_dlsbd = []
    for num_batch, batch in enumerate(eval_dataloader):
        print("Encoding batch", num_batch)
        mean_eval_ = model.encode(batch[0])
        mean_dlsbd.append(mean_eval_)
    mean_dlsbd = torch.cat(mean_dlsbd, dim=0)
    # endregion

    # region ESTIMATE DLSBD METRIC
    print("Mean dlsbd shape", mean_dlsbd.shape)
    reshaped_mean_eval = mean_dlsbd.reshape(
        (eval_dset.num_objects, -1, np.prod(list(mean_dlsbd.shape[-4:])))).detach().cpu().numpy()
    print("Reshaped mean eval shape", reshaped_mean_eval.shape)
    n_subgroups = 1
    k_values = create_combinations_omega_values_range(-10, 10, n_subgroups)
    dlsbd_values = []
    for reshaped_mean in reshaped_mean_eval:
        object_mean = np.expand_dims(reshaped_mean, axis=0)
        dlsbd_values.append(dlsbd(object_mean, k_values, be_verbose=1, factor_manifold="cylinder")[0])
    dlsbd_metric = np.mean(dlsbd_values)
    print("DLSBD values", dlsbd_values)
    print("DLSBD", dlsbd_metric)
    model_path = os.path.join(args.checkpoints_dir, args.model_name)
    np.save(f'{model_path}/dlsbd.npy', dlsbd_metric)
    if run is not None:
        run["metrics/dlsbd"].log(dlsbd_metric)
    # endregion

# region PLOT UNIQUE IMAGES
    unique_images = []
    for unique in np.unique(dset.stabs):
        unique_images.append(dset.data[dset.stabs == unique][0][0])

    unique_images = torch.tensor(np.array(unique_images), dtype=img.dtype).to(device)
    unique_embeddings = model.encode(torch.tensor(unique_images, dtype=img.dtype).to(device))


    x_rec = model.decode(unique_embeddings)
    x_rec = x_rec.permute((0, 2, 3, 1)).detach().cpu().numpy()
    for i in range(len(x_rec)):
        add_image_to_ax(1 / (1 + np.exp(-x_rec[i])))
        plt.savefig(os.path.join(save_folder, f"reconstruction_{i}.png"), bbox_inches='tight')
        add_image_to_ax(unique_images[i].permute((1, 2, 0)).detach().cpu().numpy())
        plt.savefig(os.path.join(save_folder, f"input_image_{i}.png"), bbox_inches='tight')
        if run is not None:
            run[f"reconstruction_image_{i}"].upload(plt.gcf())
# endregion




# region RECONSTRUCTIONS EMBEDDINGS ORIGINAL
unique_images = []
for unique in np.unique(dset.stabs):
    unique_images.append(dset.data[dset.stabs == unique][0][0])

unique_images = torch.tensor(np.array(unique_images), dtype=img.dtype).to(device)

unique_mean = model.encode(torch.tensor(unique_images, dtype=img.dtype).to(device))

print("Shape of unique mean", unique_mean.shape)
x_rec = model.decode(unique_mean)
x_rec = x_rec.detach().cpu().numpy()
# Plot the reconstructions in rows of 10
num_images = 10

num_orbits = len(eval_dset.data)
num_views = len(eval_dset.data[orbitnum])

print("Reconstruction shape", x_rec.shape, "Eval mean shape", eval_mean.shape, "Eval data shape", eval_dset.data.shape)
x_rec = np.moveaxis(x_rec, 1, -1)
for num_x, x in enumerate(x_rec):
    add_image_to_ax(x)
    if run is not None:
        run[f"reconstructions_"+str(num_x)].upload(plt.gcf())
plt.close("all")







# Select an image from each of the unique stabilizer objects
print("Shape of eval data", eval_dset.data.shape, eval_dset.flat_images.shape, eval_mean.shape)
x_rec = model.decode(eval_mean)
x_rec = x_rec.detach().cpu().numpy()
# Plot the reconstructions in rows of 10
num_images = 10

num_orbits = len(eval_dset.data)
num_views = len(eval_dset.data[orbitnum])

print("Reconstruction shape", x_rec.shape, "Eval mean shape", eval_mean.shape, "Eval data shape", eval_dset.data.shape)
x_rec = x_rec.reshape((num_orbits, num_views, *eval_dset.data.shape[-3:]))
x_rec = np.moveaxis(x_rec, 2, -1)
eval_mean_reshaped = eval_mean.reshape((num_orbits, num_views, eval_mean.shape[-3], eval_mean.shape[-2], eval_mean.shape[-1]))
print("Eval mean reshaped shape", eval_mean_reshaped.shape, x_rec.shape)
fig, axes = plt.subplots(3, num_views, figsize=(20, 4))
i = 0
axes[0, 0].set_ylabel("Original")
axes[1, 0].set_ylabel("Reconstruction")
axes[2, 0].set_ylabel("Filter 1")
indexes = np.arange(0, num_views)
for j in indexes:
    axes[0, j].imshow(np.moveaxis(eval_dset.data[orbitnum][j], 0, -1))
    axes[1, j].imshow(1 / (1 + np.exp(-x_rec[orbitnum][j])))
    axes[0, j].set_xticks([])
    axes[0, j].set_yticks([])
    axes[1, j].set_xticks([])
    axes[1, j].set_yticks([])
    projected_top = np.mean(np.abs(eval_mean[j, 0].detach().numpy()), keepdims=True, axis=-1)
    projected_top = np.clip(projected_top[:, :, 0], 0, np.amax(projected_top))
    axes[2, j].imshow(projected_top, cmap="Reds")
    axes[2, j].set_xticks([])
    axes[2, j].set_yticks([])
if run is not None:
    run[f"reconstructions"].upload(plt.gcf())
plt.close("all")
# endregion





# region PLOT VOXELS
fig = plt.figure(figsize=(10, 10))

num_filters = mean_numpy.shape[1]
nx, ny, nz = mean_numpy.shape[2:]
print(mean_numpy.shape)
for i in range(num_filters):
    ax = fig.add_subplot(6, 6, i + 1)
    projected_top = np.mean(np.abs(mean_numpy[0, i]), keepdims=True, axis=-1)
    projected_top = np.clip(projected_top, 0, np.amax(projected_top))
    ax.imshow(projected_top[..., -1], cmap="Reds")
    ax.set_xticks([])
    ax.set_yticks([])
if run is not None:
    run[f"voxel_filter_above"].upload(plt.gcf())

fig = plt.figure(figsize=(10, 10))
num_filters = mean_numpy.shape[1]
colors = np.ones((*mean_numpy[0, 0].shape, 4))
colors[..., 1] = np.zeros_like(mean_numpy[0, 0])
colors[..., 2] = np.zeros_like(mean_numpy[0, 0])

for i in range(num_filters):
    ax = fig.add_subplot(6, 6, i + 1, projection='3d')
    voxelarray = mean_numpy[0, i] > 0
    colors[..., 3] = mean_numpy[0, i] / (np.amax(mean_numpy[0, i]) + 1e-5)
    ax.voxels(voxelarray, facecolors=colors)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
if run is not None:
    run[f"voxel_filter_positive"].upload(plt.gcf())

for i in range(num_filters):
    ax = fig.add_subplot(6, 6, i + 1, projection='3d')
    voxelarray = mean_numpy[0, i] < 0
    colors[..., 3] = -mean_numpy[0, i] / (np.amax(mean_numpy[0, i]) + 1e-5)
    ax.voxels(voxelarray, facecolors=colors)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
if run is not None:
    run[f"voxel_filter_negative"].upload(plt.gcf())

# endregion


#     # TODO Fix latent traversal
#     # region LATENT TRAVERSAL
#     # num_points_traversal = 10
#     # if args.autoencoder != "None":
#     #     angles_traversal = np.linspace(0, 2 * np.pi, num_points_traversal, endpoint=False)
#     #     mean_traversal = torch.tensor(np.stack([np.cos(angles_traversal), np.sin(angles_traversal)], axis=-1))
#     #     if args.extra_dim > 0:
#     #
#     #         print(extra_eval.shape)
#     #         mean_extra = extra_eval.mean(dim=0)
#     #         print(mean_extra.shape)
#     #         mean_extra = mean_extra.unsqueeze(0)
#     #         print(mean_extra.shape)
#     #         mean_extra = mean_extra.repeat((num_points_traversal,1), 0)
#     #         # mean_extra = mean_extra.repeat(num_points_traversal, 0)
#     #         z = torch.cat([mean_traversal, mean_extra], dim=-1).float()
#     #         print(z.shape, mean_traversal.shape, mean_extra.shape)
#     #         x_rec = decoder(z)
#     #     else:
#     #         x_rec = decoder(mean_traversal)
#     #     fig, axes = plt.subplots(1, num_points_traversal, figsize=(num_points_traversal, 1))
#     #     for i in range(num_points_traversal):
#     #         axes[i].imshow(x_rec[i].permute((1, 2, 0)).detach().cpu().numpy())
#     #         axes[i].axis('off')
#     #     fig.savefig(os.path.join(save_folder, f"traversal.png"), bbox_inches='tight')
#     # endregion
#
#     # region ESTIMATE DLSBD METRIC
#     print("Estimating DLSBD metric")
#     # reshaped_eval_actions_gaussians = repeat_angles_n_gaussians(eval_dset.flat_lbls, N)
#     # reshaped_mean_eval = mean_eval.reshape(eval_dset.num_objects, -1, N, args.latent_dim)
#     # z_inv = apply_inverse_rotation(reshaped_mean_eval.detach().numpy(), reshaped_eval_actions_gaussians)
#     #
#     # # TODO: Clean this computation
#     # if args.latent_dim == 4:
#     #     reshaped_eval_actions = eval_dset.flat_lbls.reshape(eval_dset.num_objects, -1, 2)
#     #     fig, axes = plt.subplots(1, 2, figsize=(10, 5))
#     #     for num_subspace in range(2):
#     #         mean_subspace = (mean_eval[..., num_subspace * 2: (num_subspace + 1) * 2]).reshape((-1, 2))
#     #         z_inv_subspace = (z_inv[..., num_subspace * 2: (num_subspace + 1) * 2]).reshape((-1, 2))
#     #         axes[num_subspace].scatter(mean_subspace[:, 0].detach().numpy(), mean_subspace[:, 1].detach().numpy())
#     #         axes[num_subspace].scatter(z_inv_subspace[:, 0], z_inv_subspace[:, 1], marker="*")
#     #         for num_object in range(eval_dset.num_objects):
#     #             z_inv_mean = estimate_kmeans_inv(z_inv, eval_dset.stabs[num_object, 0, :])
#     #             z_mean_inv_subspace = z_inv_mean[num_subspace]
#     #             axes[num_subspace].scatter(z_mean_inv_subspace[:, 0], z_mean_inv_subspace[:, 1], marker="*", c="r")
#     #         axes[num_subspace].set_title(f"Subspace {num_subspace}")
#     #
#     # else:
#     #     reshaped_eval_actions = eval_dset.flat_lbls.reshape(eval_dset.num_objects, -1)
#     #     fig = plt.figure()
#     #     mean_eval_flat = mean_eval.reshape((-1, args.latent_dim))
#     #     plt.scatter(mean_eval_flat[:, 0].detach().numpy(), mean_eval_flat[:, 1].detach().numpy())
#     #     z_inv_flat = z_inv.reshape((-1, args.latent_dim))
#     #     plt.scatter(z_inv_flat[:, 0], z_inv_flat[:, 1], marker="*")
#     #     for num_object in range(eval_dset.num_objects):
#     #         z_inv_mean = estimate_kmeans_inv(z_inv, eval_dset.flat_stabs[num_object, 0])[0]
#     #         plt.scatter(z_inv_mean[:, 0], z_inv_mean[:, 1], marker="*", c="r")
#     #
#     # plt.savefig(os.path.join(save_folder, f"z_inv.png"), bbox_inches='tight')
#     # dlsbd = dlsbd_metric_mixture(reshaped_mean_eval.detach().numpy(), reshaped_eval_actions, eval_dset.flat_stabs[:, 0], False,
#     #                              distance_function="chamfer")
#     # dlsbd = np.mean(dlsbd)
#     # print("DLSBD METRIC!!!",
#     #       dlsbd_metric_mixture(reshaped_mean_eval.detach().numpy(), reshaped_eval_actions, eval_dset.flat_stabs[:, 0], True,
#     #                            distance_function="chamfer"))
#     # np.save(os.path.join(save_folder, f"dlsbd.npy"), dlsbd)
#     #
#     # with open(os.path.join(save_folder, "metrics.json"), "w") as f:
#     #     json.dump({"dlsbd": dlsbd}, f)
#     # endregion
#
# # Plotting for SO(3)
# elif args.latent_dim == 3:
#     for i in range(10):
#         # plt.figure()
#         # plt.imshow(npimages_eval[i])
#         # plt.figure()
#         # ax = plt.axes(projection='3d')
#         # rots = R.from_dcm(mean[i]).as_rotvec()  # .as_euler('zxy', degrees=False)
#         # ax.scatter3D(rots[:, 0], rots[:, 1], rots[:, 2], s=[30] * len(rots))
#         # plt.show()
#
#         mean_rot = (action @ mean).detach().cpu().numpy()
#         fig = plt.figure(figsize=(10, 10))
#
#         ax = plt.subplot(221)
#         ax.imshow(npimages[i])
#         ax.set_xticks([])
#         ax.set_yticks([])
#         ax = plt.subplot(222)
#         ax.imshow(npimages_next[i])
#         ax.set_xticks([])
#         ax.set_yticks([])
#         ax = plt.subplot(223, projection='mollweide')
#
#         _ = visualize_so3_probabilities(mean_numpy[i], 0.05 * np.ones(len(mean_numpy[i])), ax=ax, fig=fig,
#                                         show_color_wheel=True)
#         ax = plt.subplot(224, projection='mollweide')
#         _ = visualize_so3_probabilities(mean_next[i], 0.05 * np.ones(len(mean_next[i])), ax=ax, fig=fig,
#                                         rotations_gt=mean_rot[i],
#                                         show_color_wheel=True)
#         fig.savefig(os.path.join(save_folder, f"{i}.png"), bbox_inches='tight')
#         plt.close("all")
#
#         fig = plt.figure(figsize=(10, 10))
#         ax = plt.subplot(221)
#         ax.imshow(npimages[i])
#         ax = plt.subplot(222)
#         ax.imshow(npimages_next[i])
#         ax = plt.subplot(223, projection='3d')
#         rots = R.from_matrix(mean_numpy[i]).as_rotvec()  # .as_euler('zxy', degrees=False)
#         ax.scatter3D(rots[:, 0], rots[:, 1], rots[:, 2], s=[30] * len(rots))
#         ax = plt.subplot(224, projection='3d')
#         rots = R.from_matrix(mean_next[i]).as_rotvec()  # .as_euler('zxy', degrees=False)
#         ax.scatter3D(rots[:, 0], rots[:, 1], rots[:, 2], s=[30] * len(rots), c="r", alpha=0.1)
#         rots = R.from_matrix(mean_rot[i]).as_rotvec()  # .as_euler('zxy', degrees=False)
#         ax.scatter3D(rots[:, 0], rots[:, 1], rots[:, 2], s=[30] * len(rots), marker="*")
#         fig.savefig(os.path.join(save_folder, f"rotvec_{i}.png"), bbox_inches='tight')
#         plt.close("all")
#
#         fig = plt.figure(figsize=(5, 5))
#         ax = plt.subplot(111)
#         add_image_to_ax(npimages_next[i], ax=ax)
#         fig.savefig(os.path.join(save_folder, f"image_alone_{i}.png"), bbox_inches='tight')
#         plt.close("all")
#         fig = plt.figure(figsize=(5, 5))
#         ax = plt.subplot(111, projection='3d')
#         ax.scatter3D(rots[:, 0], rots[:, 1], rots[:, 2], s=[30] * len(rots), c="r", alpha=0.5)
#         N = 200
#         stride = 1
#         u = np.linspace(0, 2 * np.pi, N)
#         v = np.linspace(0, np.pi, N)
#         radius = np.sqrt(np.pi)
#         x = np.outer(np.cos(u) * radius, np.sin(v) * radius)
#         y = np.outer(np.sin(u) * radius, np.sin(v) * radius)
#         z = np.outer(np.ones(np.size(u)) * radius, np.cos(v) * radius)
#         ax.plot_surface(x, y, z, linewidth=0.0, cstride=stride, alpha=0.05, color="k", rstride=stride)
#         ax.set_xticks([])
#         ax.set_yticks([])
#         ax.set_zticks([])
#         fig.savefig(os.path.join(save_folder, f"rotvec_alone_{i}.png"), bbox_inches='tight')
#

#     # region PLOT RECONSTRUCTIONS
#     if args.autoencoder != "None":
#         x_rec = decoder(torch.cat([unique_mean.view((unique_mean.shape[0], -1)), unique_extra], dim=-1))
#         x_rec = x_rec.permute((0, 2, 3, 1)).detach().cpu().numpy()
#         for i in range(len(x_rec)):
#             add_image_to_ax(1 / (1 + np.exp(-x_rec[i])))
#             plt.savefig(os.path.join(save_folder, f"reconstruction_{i}.png"), bbox_inches='tight')
#             add_image_to_ax(unique_images[i].permute((1, 2, 0)).detach().cpu().numpy())
#             plt.savefig(os.path.join(save_folder, f"input_image_{i}.png"), bbox_inches='tight')
if run is not None:
    run.stop()
