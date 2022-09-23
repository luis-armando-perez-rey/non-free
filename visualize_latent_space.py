import argparse
import json
from scipy.spatial.transform import Rotation as R

import matplotlib.pyplot as plt
import torch.utils.data
from torch import load

# Import datasets
from datasets.equiv_dset import *
from utils.nn_utils import *

# Import plotting utils
from utils.plotting_utils import plot_extra_dims, plot_images_distributions, plot_embeddings_eval, \
    save_embeddings_on_circle, plot_images_multi_reconstructions, load_plot_val_errors, \
    plot_embeddings_eval_torus, yiq_embedding, plot_mixture_neurreps, add_image_to_ax, add_distribution_to_ax_torus
from utils.disentanglement_metric import dlsbd_metric_mixture, repeat_angles_n_gaussians, apply_inverse_rotation, \
    estimate_mean_inv, estimate_kmeans_inv
from utils.plotting_utils_so3 import visualize_so3_probabilities
from models.losses import ReconstructionLoss

parser = argparse.ArgumentParser()
parser.add_argument('--save-folder', type=str, default='checkpoints', help='Path to saved model')
parser.add_argument('--dataset', type=str, default='dataset', help='Dataset')
parser.add_argument('--dataset_name', nargs="+", type=str, default=['4'], help='Dataset name')
args_eval = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.manual_seed(42)
torch.cuda.empty_cache()

model_dir = os.path.join(".", "saved_models", args_eval.save_folder)
model_file = os.path.join(model_dir, 'model.pt')
decoder_file = os.path.join(model_dir, 'decoder.pt')
meta_file = os.path.join(model_dir, 'metadata.pkl')
args = pickle.load(open(meta_file, 'rb'))['args']
N = args.num
save_folder = os.path.join(".", "visualizations", args.model_name)
os.makedirs(save_folder, exist_ok=True)
print(args)
if args.dataset == 'square':
    dset = EquivDataset(f'{args.data_dir}/square/', list_dataset_names=args.dataset_name)
    flat_stabilizers = None
    stabilizers = None
    eval_images = None
elif args.dataset == 'platonics':
    dset = PlatonicMerged(N=30000, data_dir=args.data_dir)
    stabilizers = None
elif args.dataset == "symmetric_solids":
    dset = EquivDatasetStabs(f'{args.data_dir}/symmetric_solids/', list_dataset_names=args.dataset_name)
    stabilizers = None
    eval_images = None
    flat_stabilizers = None
elif args.dataset == "arrows" or args.dataset == "sinusoidal":
    dset = EquivDatasetStabs(f'{args.data_dir}/{args.dataset}/', list_dataset_names=args.dataset_name)
    dset_eval = EvalDataset(f'{args.data_dir}/{args.dataset}/', list_dataset_names=args.dataset_name)
    num_objects = dset_eval.data.shape[0]
    eval_images = torch.FloatTensor(dset_eval.data.reshape(-1, *dset_eval.data.shape[2:]))
    stabilizers = dset_eval.stabs
    flat_stabilizers = dset_eval.stabs.reshape((-1))
    flat_eval_actions = dset_eval.lbls.reshape((-1))
    eval_actions = dset_eval.lbls

elif args.dataset.endswith("translation"):
    dset = EquivDatasetStabs(f'{args.data_dir}/{args.dataset}/', list_dataset_names=args.dataset_name)
    dset_eval = EvalDataset(f'{args.data_dir}/{args.dataset}/', list_dataset_names=args.dataset_name)
    eval_images = torch.FloatTensor(dset_eval.data.reshape(-1, *dset_eval.data.shape[2:]))
    num_objects = dset_eval.data.shape[0]
    stabilizers = dset_eval.stabs
    flat_stabilizers = dset_eval.stabs.reshape((-1, 2))
    flat_eval_actions = dset_eval.lbls.reshape((-1, 2))
    eval_actions = dset_eval.lbls
    print("Number of objects", num_objects)
elif args.dataset == "double_arrows":
    dset = EquivDatasetStabs(f'{args.data_dir}/{args.dataset}/', list_dataset_names=args.dataset_name)
    dset_eval = EvalDataset(f'{args.data_dir}/{args.dataset}/', list_dataset_names=args.dataset_name)
    eval_images = torch.FloatTensor(dset_eval.data.reshape(-1, *dset_eval.data.shape[2:]))
    num_objects = dset_eval.data.shape[0]
    stabilizers = dset_eval.stabs
    flat_stabilizers = dset_eval.stabs.reshape((-1, 2))
    flat_eval_actions = dset_eval.lbls.reshape((-1, 2))
    eval_actions = dset_eval.lbls
    print("Number of objects", num_objects)
else:
    eval_images = None
    flat_stabilizers = None
    raise ValueError(f'Dataset {args.dataset} not supported')

train_loader = torch.utils.data.DataLoader(dset,
                                           batch_size=100, shuffle=True)
device = 'cpu'
model = load(model_file).to(device)
if args.autoencoder != 'None':
    decoder = load(decoder_file).to(device)
model.eval()

if args.dataset == "arrows" or args.dataset == "sinusoidal" or args.dataset.endswith(
        "translation") or args.dataset == "double_arrows":
    img, img_next, action, n_stabilizers = next(iter(train_loader))
    print("EVAL IMAGES SHAPE", eval_images.shape)
    mean_eval, logvar_eval, extra_eval = model(eval_images.to(device))
elif args.dataset == "symmetric_solids":
    img, img_next, action, n_stabilizers = next(iter(train_loader))
    eval_images = img
    mean_eval, logvar_eval, extra_eval = model(eval_images.to(device))
elif args.dataset == "platonics":
    img, img_next, action = next(iter(train_loader))
    eval_images = img
    mean_eval, logvar_eval, extra_eval = model(eval_images.to(device))

else:
    img = None
    img_next = None
    action = None
    n_stabilizers = None
    logvar_eval = None
    std_eval = None
    ValueError(f"Dataset {args.dataset} not implemented for visualization")

# Plot the training evaluation
fig, _ = load_plot_val_errors(os.path.join(model_dir, "errors_val.npy"))
fig.savefig(os.path.join(save_folder, 'erros_val.png'))

img_shape = np.array(img.shape[1:])
if img.dim() == 2:
    # When the data is not composed of images
    npimages = img.detach().cpu().numpy()
    npimages_next = img_next.detach().cpu().numpy()
    npimages_eval = eval_images.detach().cpu().numpy()
else:
    # When data is composed of images
    npimages = np.squeeze(np.transpose(img.detach().cpu().numpy(), axes=[0, 2, 3, 1]))
    npimages_next = np.squeeze(np.transpose(img_next.detach().cpu().numpy(), axes=[0, 2, 3, 1]))
    npimages_eval = np.squeeze(np.transpose(eval_images.detach().cpu().numpy(), axes=[0, 2, 3, 1]))

mean, logvar, extra = model(img.to(device))
mean_next, logvar_next, extra_next = model(img_next.to(device))
logvar = -4.6 * torch.ones(logvar.shape).to(logvar.device)
logvar_next = -4.6 * torch.ones(logvar.shape).to(logvar.device)
logvar_eval = -4.6 * torch.ones(logvar_eval.shape).to(logvar_eval.device)
std_eval = np.exp(logvar_eval.detach().cpu().numpy() / 2.) / 10

mean_numpy = mean.detach().cpu().numpy()
mean_next = mean_next.detach().cpu().numpy()
std = np.exp(logvar.detach().cpu().numpy() / 2.) / 10
std_next = np.exp(logvar_next.detach().cpu().numpy() / 2.) / 10

if args.latent_dim == 2 or args.latent_dim == 4:
    if args.latent_dim == 2:
        action = action.squeeze(1)
        rot = make_rotation_matrix(action)
        mean_rot = (rot @ mean.unsqueeze(-1)).squeeze(-1)
        mean_rot = mean_rot.detach().cpu().numpy()
    else:
        action = action.squeeze(1)
        mean_rot = so2_rotate_subspaces(mean, action, detach=True)

    action = action.detach().cpu().numpy()

    # TODO: Review plot identity embeddings code
    # region PLOT IDENTITY EMBEDDINGS
    # if (args.latent_dim != 4) and (args.extra_dim > 0) :
    #     fig, ax = plot_extra_dims(extra, color_labels=flat_stabilizers)
    #     if fig:
    #         fig.savefig(os.path.join(save_folder, 'invariant.png'))
    # endregion

    # region PLOT ROTATED EMBEDDINGS
    for i in range(10):
        print(f"Plotting example {i}")
        print(mean_numpy.shape)
        fig, axes = plot_images_distributions(mean=mean_numpy[i], std=std[i], mean_next=mean_next[i],
                                              std_next=std_next[i],
                                              image=npimages[i], image_next=npimages_next[i],
                                              expected_mean=mean_rot[i], n=N)
        plt.savefig(os.path.join(save_folder, f"image_pair_{i}.png"), bbox_inches='tight')
        plot_mixture_neurreps(mean_numpy[i])
        plt.savefig(os.path.join(save_folder, f"test_mixture_{i}.png"), bbox_inches='tight')
        add_image_to_ax(npimages[i])
        plt.savefig(os.path.join(save_folder, f"test_image_{i}.png"), bbox_inches='tight')
        if args.latent_dim == 4:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax = add_distribution_to_ax_torus(mean_numpy[i], std[i], ax, n=N, color="r", scatter=True)
            ax.set_xlim([-np.pi, np.pi])
            ax.set_ylim([-np.pi, np.pi])
            ax.set_xticks([])
            ax.set_yticks([])

            plt.savefig(os.path.join(save_folder, f"test_mixture_{i}.png"), bbox_inches='tight')
            add_image_to_ax(npimages[i])
            plt.savefig(os.path.join(save_folder, f"test_image_{i}.png"), bbox_inches='tight')

        # fig.savefig(os.path.join(save_folder, f"test_{i}.png"), bbox_inches='tight')
        plt.close("all")
    # # endregion
    # # Save the plots of the embeddings on the circle
    # save_embeddings_on_circle(mean_eval, std_eval, flat_stabilizers, save_folder, args.dataset_name[0],
    #                           increasing_radius=False)
    #
    # save_embeddings_on_circle(mean_eval, std_eval, flat_stabilizers, save_folder, args.dataset_name[0],
    #                           increasing_radius=True)
    #
    # # region PLOT ON THE TORUS
    # if args.latent_dim == 4:
    #     print("EVAL ACTIONS SHAPE", flat_eval_actions.shape)
    #     colors = yiq_embedding(flat_eval_actions[:, 0], flat_eval_actions[:, 1])
    #     fig, ax = plot_embeddings_eval_torus(mean_eval.detach().numpy(), colors)
    #     fig.savefig(os.path.join(save_folder, f"torus_eval_embedings.png"), bbox_inches='tight')
    #     plt.close("all")
    # endregion

    # TODO: Improve reconstruction code
    unique_images = []
    for unique in np.unique(dset.stabs):
        unique_images.append(dset.data[dset.stabs == unique][0][0])

    unique_images = torch.tensor(np.array(unique_images), dtype=img.dtype).to(device)
    unique_mean, unique_logvar, unique_extra = model(unique_images)
    # region PLOT RECONSTRUCTIONS
    if args.autoencoder != "None":
        x_rec = decoder(torch.cat([unique_mean.view((unique_mean.shape[0], -1)), unique_extra], dim=-1))
        x_rec = x_rec.permute((0, 2, 3, 1)).detach().cpu().numpy()
        for i in range(len(x_rec)):
            add_image_to_ax(1 / (1 + np.exp(-x_rec[i])))
            plt.savefig(os.path.join(save_folder, f"reconstruction_{i}.png"), bbox_inches='tight')
            add_image_to_ax(unique_images[i].permute((1, 2, 0)).detach().cpu().numpy())
            plt.savefig(os.path.join(save_folder, f"input_image_{i}.png"), bbox_inches='tight')

            # boolean_selection = (flat_stabilizers == unique)
            # fig, _ = plot_images_multi_reconstructions(npimages_eval[boolean_selection][:5],
            #                                            reconstructions_np[boolean_selection][:5])
            # fig.savefig(os.path.join(save_folder, f"{unique}_reconstructions.png"), bbox_inches='tight')
    # endregion

    # TODO Fix latent traversal
    # region LATENT TRAVERSAL
    # num_points_traversal = 10
    # if args.autoencoder != "None":
    #     angles_traversal = np.linspace(0, 2 * np.pi, num_points_traversal, endpoint=False)
    #     mean_traversal = torch.tensor(np.stack([np.cos(angles_traversal), np.sin(angles_traversal)], axis=-1))
    #     if args.extra_dim > 0:
    #
    #         print(extra_eval.shape)
    #         mean_extra = extra_eval.mean(dim=0)
    #         print(mean_extra.shape)
    #         mean_extra = mean_extra.unsqueeze(0)
    #         print(mean_extra.shape)
    #         mean_extra = mean_extra.repeat((num_points_traversal,1), 0)
    #         # mean_extra = mean_extra.repeat(num_points_traversal, 0)
    #         z = torch.cat([mean_traversal, mean_extra], dim=-1).float()
    #         print(z.shape, mean_traversal.shape, mean_extra.shape)
    #         x_rec = decoder(z)
    #     else:
    #         x_rec = decoder(mean_traversal)
    #     fig, axes = plt.subplots(1, num_points_traversal, figsize=(num_points_traversal, 1))
    #     for i in range(num_points_traversal):
    #         axes[i].imshow(x_rec[i].permute((1, 2, 0)).detach().cpu().numpy())
    #         axes[i].axis('off')
    #     fig.savefig(os.path.join(save_folder, f"traversal.png"), bbox_inches='tight')
    # endregion

    # region ESTIMATE DLSBD METRIC
    print("Estimating DLSBD metric")
    reshaped_eval_actions_gaussians = repeat_angles_n_gaussians(eval_actions, N)
    reshaped_mean_eval = mean_eval.reshape(num_objects, -1, N, args.latent_dim)
    z_inv = apply_inverse_rotation(reshaped_mean_eval.detach().numpy(), reshaped_eval_actions_gaussians)

    # TODO: Clean this computation
    if args.latent_dim == 4:
        reshaped_eval_actions = flat_eval_actions.reshape(num_objects, -1, 2)
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        for num_subspace in range(2):
            mean_subspace = (mean_eval[..., num_subspace * 2: (num_subspace + 1) * 2]).reshape((-1, 2))
            z_inv_subspace = (z_inv[..., num_subspace * 2: (num_subspace + 1) * 2]).reshape((-1, 2))
            axes[num_subspace].scatter(mean_subspace[:, 0].detach().numpy(), mean_subspace[:, 1].detach().numpy())
            axes[num_subspace].scatter(z_inv_subspace[:, 0], z_inv_subspace[:, 1], marker="*")
            for num_object in range(num_objects):
                z_inv_mean = estimate_kmeans_inv(z_inv, stabilizers[num_object, 0, :])
                z_mean_inv_subspace = z_inv_mean[num_subspace]
                axes[num_subspace].scatter(z_mean_inv_subspace[:, 0], z_mean_inv_subspace[:, 1], marker="*", c="r")
            axes[num_subspace].set_title(f"Subspace {num_subspace}")

    else:
        reshaped_eval_actions = flat_eval_actions.reshape(num_objects, -1)
        fig = plt.figure()
        mean_eval_flat = mean_eval.reshape((-1, args.latent_dim))
        plt.scatter(mean_eval_flat[:, 0].detach().numpy(), mean_eval_flat[:, 1].detach().numpy())
        z_inv_flat = z_inv.reshape((-1, args.latent_dim))
        plt.scatter(z_inv_flat[:, 0], z_inv_flat[:, 1], marker="*")
        for num_object in range(num_objects):
            z_inv_mean = estimate_kmeans_inv(z_inv, stabilizers[num_object, 0])[0]
            plt.scatter(z_inv_mean[:, 0], z_inv_mean[:, 1], marker="*", c="r")

    plt.savefig(os.path.join(save_folder, f"z_inv.png"), bbox_inches='tight')
    dlsbd = dlsbd_metric_mixture(reshaped_mean_eval.detach().numpy(), reshaped_eval_actions, stabilizers[:, 0], False,
                                 distance_function="chamfer")
    dlsbd = np.mean(dlsbd)
    print("DLSBD METRIC!!!",
          dlsbd_metric_mixture(reshaped_mean_eval.detach().numpy(), reshaped_eval_actions, stabilizers[:, 0], True,
                               distance_function="chamfer"))
    np.save(os.path.join(save_folder, f"dlsbd.npy"), dlsbd)

    with open(os.path.join(save_folder, "metrics.json"), "w") as f:
        json.dump({"dlsbd": dlsbd}, f)
    # endregion

elif args.latent_dim == 3:
    for i in range(10):
        # plt.figure()
        # plt.imshow(npimages_eval[i])
        # plt.figure()
        # ax = plt.axes(projection='3d')
        # rots = R.from_dcm(mean[i]).as_rotvec()  # .as_euler('zxy', degrees=False)
        # ax.scatter3D(rots[:, 0], rots[:, 1], rots[:, 2], s=[30] * len(rots))
        # plt.show()
        fig = plt.figure(figsize=(10, 10))

        ax = plt.subplot(121)
        ax.imshow(npimages_eval[i])
        ax = plt.subplot(122, projection='mollweide')
        _ = visualize_so3_probabilities(mean[i].detach().numpy(), 0.05 * np.ones(len(mean[i])), ax=ax, fig=fig,
                                        show_color_wheel=True)
        fig.savefig(os.path.join(save_folder, f"{i}.png"), bbox_inches='tight')
        plt.close("all")

        fig = plt.figure(figsize=(10, 5))
        ax = plt.subplot(121)
        ax.imshow(npimages_eval[i])
        ax = plt.subplot(122, projection='3d')
        rots = R.from_matrix(mean_numpy[i]).as_rotvec()  # .as_euler('zxy', degrees=False)
        ax.scatter3D(rots[:, 0], rots[:, 1], rots[:, 2], s=[30] * len(rots))
        fig.savefig(os.path.join(save_folder, f"rotvec_{i}.png"), bbox_inches='tight')
        plt.close("all")
