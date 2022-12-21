import argparse
import pickle
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from utils.dataset_utils import get_dataset
from utils.nn_utils import get_rotated_mean
from utils.plotting_utils import plot_extra_dims, plot_images_distributions, \
    plot_mixture_neurreps, add_image_to_ax, add_distribution_to_ax_torus, save_embeddings_on_circle
from utils.plotting_utils_so3 import visualize_so3_probabilities
from utils.torch_utils import torch_data_to_numpy

parser = argparse.ArgumentParser()
parser.add_argument('--save-folder', type=str, default='checkpoints', help='Path to saved model')
parser.add_argument('--dataset', type=str, default='dataset', help='Dataset')
parser.add_argument('--dataset_name', nargs="+", type=str, default=['4'], help='Dataset name')
args_eval = parser.parse_args()



model_dir = os.path.join(".", "saved_models", args_eval.save_folder)
model_file = os.path.join(model_dir, 'model.pt')
decoder_file = os.path.join(model_dir, 'decoder.pt')
meta_file = os.path.join(model_dir, 'metadata.pkl')
args = pickle.load(open(meta_file, 'rb'))['args']
device = torch.device("cuda:"+args.gpu if torch.cuda.is_available() else "cpu")
# torch.manual_seed(42)
torch.cuda.empty_cache()
print(args)
save_folder = os.path.join(".", "visualizations", args.model_name)
os.makedirs(save_folder, exist_ok=True)


if args.neptune_user != "":
    from utils.neptune_utils import reload_neptune_run

    neptune_id_file = os.path.join(model_dir, 'neptune.txt')
    run = reload_neptune_run(args.neptune_user, "non-free", neptune_id_file)
else:
    run = None


# region LOAD DATASET
dset, eval_dset = get_dataset(args.data_dir, args.dataset, args.dataset_name)
train_loader = torch.utils.data.DataLoader(dset, batch_size=100, shuffle=True)
# endregion

# region GET MODEL
device = 'cpu'
model = torch.load(model_file).to(device)
if args.autoencoder != 'None':
    decoder = torch.load(decoder_file).to(device)
else:
    decoder = None
model.eval()
# endregion

# region GET IMAGES
img, img_next, action, n_stabilizers = next(iter(train_loader))
mean_eval, logvar_eval, extra_eval = model(torch.FloatTensor(eval_dset.flat_images).to(device))
img_shape = np.array(img.shape[1:])

# Get the numpy array versions of the images
npimages = torch_data_to_numpy(img)
npimages_next = torch_data_to_numpy(img_next)
npimages_eval = eval_dset.flat_images_numpy
# endregion

# region POSTERIOR PARAMETERS
# Calculate the parameters obtained by the models
mean, logvar, extra = model(img.to(device))
extra = extra.detach().cpu().numpy()
mean_next, logvar_next, extra_next = model(img_next.to(device))
extra_next = extra_next.detach().cpu().numpy()
logvar = -4.6 * torch.ones(logvar.shape).to(logvar.device)
logvar_eval = -4.6 * torch.ones(logvar_eval.shape).to(logvar_eval.device)
std_eval = np.exp(logvar_eval.detach().cpu().numpy() / 2.) / 10

# Obtain the values as numpy arrays
mean_numpy = mean.detach().cpu().numpy()
mean_next = mean_next.detach().cpu().numpy()
std = np.exp(logvar.detach().cpu().numpy() / 2.) / 10
std_next = np.exp(logvar_next.detach().cpu().numpy() / 2.) / 10
# endregion



# Plotting for SO(2) and its combinations
if args.latent_dim == 2 or args.latent_dim == 4:
    action = action.squeeze(1)
    mean_rot = get_rotated_mean(mean, action, args.latent_dim)
    action = action.detach().cpu().numpy()

    # region PLOT IDENTITY EMBEDDINGS
    if (args.latent_dim != 4) and (args.extra_dim > 0):
        fig, ax = plot_extra_dims(extra)
        if fig:
            fig.savefig(os.path.join(save_folder, 'invariant.png'))
            if run is not None:
                run["invariant"].upload(fig)
    # endregion

    # region PLOT ROTATED EMBEDDINGS
    for i in range(10):
        print(f"Plotting example {i}")
        fig, axes = plot_images_distributions(mean=mean_numpy[i], std=std[i], mean_next=mean_next[i],
                                              std_next=std_next[i],
                                              image=npimages[i], image_next=npimages_next[i],
                                              expected_mean=mean_rot[i], n=args.num)
        plt.savefig(os.path.join(save_folder, f"image_pair_{i}.png"), bbox_inches='tight')
        if run is not None:
            run[f"image_pair_{i}"].upload(fig)
        plot_mixture_neurreps(mean_numpy[i])
        if run is not None:
            run[f"test_mixture_{i}"].upload(fig)
        plt.savefig(os.path.join(save_folder, f"test_mixture_{i}.png"), bbox_inches='tight')
        add_image_to_ax(npimages[i])
        if run is not None:
            run[f"image_{i}"].upload(fig)
        plt.savefig(os.path.join(save_folder, f"test_image_{i}.png"), bbox_inches='tight')
        if args.latent_dim == 4:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax = add_distribution_to_ax_torus(mean_numpy[i], std[i], ax, n=args.num, color="r", scatter=True)
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

    save_embeddings_on_circle(mean_eval.detach().cpu().numpy(), std_eval, eval_dset.flat_stabs, save_folder,
                              args.dataset_name[0],
                              increasing_radius=True)
    if run is not None:
        run["embeddings_on_circle"].upload(plt.gcf())
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
        if args.autoencoder == "ae_single":
            print(unique_mean.shape, unique_extra.shape)
            x_rec = decoder(torch.cat([unique_mean[:, 0], unique_extra], dim=-1))
            x_rec = x_rec.permute((0, 2, 3, 1)).detach().cpu().numpy()
            for i in range(len(x_rec)):
                add_image_to_ax(1 / (1 + np.exp(-x_rec[i])))
                if run is not None:
                    run[f"reconstruction_{i}"].upload(plt.gcf())
                plt.savefig(os.path.join(save_folder, f"reconstruction_{i}.png"), bbox_inches='tight')
                add_image_to_ax(unique_images[i].permute((1, 2, 0)).detach().cpu().numpy())
                plt.savefig(os.path.join(save_folder, f"input_image_{i}.png"), bbox_inches='tight')
        else:
            x_rec = decoder(torch.cat([unique_mean.view((unique_mean.shape[0], -1)), unique_extra], dim=-1))
            x_rec = x_rec.permute((0, 2, 3, 1)).detach().cpu().numpy()
            for i in range(len(x_rec)):
                add_image_to_ax(1 / (1 + np.exp(-x_rec[i])))
                if run is not None:
                    run[f"reconstruction_{i}"].upload(plt.gcf())
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
    # reshaped_eval_actions_gaussians = repeat_angles_n_gaussians(eval_dset.flat_lbls, N)
    # reshaped_mean_eval = mean_eval.reshape(eval_dset.num_objects, -1, N, args.latent_dim)
    # z_inv = apply_inverse_rotation(reshaped_mean_eval.detach().numpy(), reshaped_eval_actions_gaussians)
    #
    # # TODO: Clean this computation
    # if args.latent_dim == 4:
    #     reshaped_eval_actions = eval_dset.flat_lbls.reshape(eval_dset.num_objects, -1, 2)
    #     fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    #     for num_subspace in range(2):
    #         mean_subspace = (mean_eval[..., num_subspace * 2: (num_subspace + 1) * 2]).reshape((-1, 2))
    #         z_inv_subspace = (z_inv[..., num_subspace * 2: (num_subspace + 1) * 2]).reshape((-1, 2))
    #         axes[num_subspace].scatter(mean_subspace[:, 0].detach().numpy(), mean_subspace[:, 1].detach().numpy())
    #         axes[num_subspace].scatter(z_inv_subspace[:, 0], z_inv_subspace[:, 1], marker="*")
    #         for num_object in range(eval_dset.num_objects):
    #             z_inv_mean = estimate_kmeans_inv(z_inv, eval_dset.stabs[num_object, 0, :])
    #             z_mean_inv_subspace = z_inv_mean[num_subspace]
    #             axes[num_subspace].scatter(z_mean_inv_subspace[:, 0], z_mean_inv_subspace[:, 1], marker="*", c="r")
    #         axes[num_subspace].set_title(f"Subspace {num_subspace}")
    #
    # else:
    #     reshaped_eval_actions = eval_dset.flat_lbls.reshape(eval_dset.num_objects, -1)
    #     fig = plt.figure()
    #     mean_eval_flat = mean_eval.reshape((-1, args.latent_dim))
    #     plt.scatter(mean_eval_flat[:, 0].detach().numpy(), mean_eval_flat[:, 1].detach().numpy())
    #     z_inv_flat = z_inv.reshape((-1, args.latent_dim))
    #     plt.scatter(z_inv_flat[:, 0], z_inv_flat[:, 1], marker="*")
    #     for num_object in range(eval_dset.num_objects):
    #         z_inv_mean = estimate_kmeans_inv(z_inv, eval_dset.flat_stabs[num_object, 0])[0]
    #         plt.scatter(z_inv_mean[:, 0], z_inv_mean[:, 1], marker="*", c="r")
    #
    # plt.savefig(os.path.join(save_folder, f"z_inv.png"), bbox_inches='tight')
    # dlsbd = dlsbd_metric_mixture(reshaped_mean_eval.detach().numpy(), reshaped_eval_actions, eval_dset.flat_stabs[:, 0], False,
    #                              distance_function="chamfer")
    # dlsbd = np.mean(dlsbd)
    # print("DLSBD METRIC!!!",
    #       dlsbd_metric_mixture(reshaped_mean_eval.detach().numpy(), reshaped_eval_actions, eval_dset.flat_stabs[:, 0], True,
    #                            distance_function="chamfer"))
    # np.save(os.path.join(save_folder, f"dlsbd.npy"), dlsbd)
    #
    # with open(os.path.join(save_folder, "metrics.json"), "w") as f:
    #     json.dump({"dlsbd": dlsbd}, f)
    # endregion

# Plotting for SO(3)
elif args.latent_dim == 3:
    for i in range(10):
        # plt.figure()
        # plt.imshow(npimages_eval[i])
        # plt.figure()
        # ax = plt.axes(projection='3d')
        # rots = R.from_dcm(mean[i]).as_rotvec()  # .as_euler('zxy', degrees=False)
        # ax.scatter3D(rots[:, 0], rots[:, 1], rots[:, 2], s=[30] * len(rots))
        # plt.show()

        mean_rot = (action @ mean).detach().cpu().numpy()
        fig = plt.figure(figsize=(10, 10))

        ax = plt.subplot(221)
        ax.imshow(npimages[i])
        ax.set_xticks([])
        ax.set_yticks([])
        ax = plt.subplot(222)
        ax.imshow(npimages_next[i])
        ax.set_xticks([])
        ax.set_yticks([])
        ax = plt.subplot(223, projection='mollweide')

        _ = visualize_so3_probabilities(mean_numpy[i], 0.05 * np.ones(len(mean_numpy[i])), ax=ax, fig=fig,
                                        show_color_wheel=True)
        ax = plt.subplot(224, projection='mollweide')
        _ = visualize_so3_probabilities(mean_next[i], 0.05 * np.ones(len(mean_next[i])), ax=ax, fig=fig,
                                        rotations_gt=mean_rot[i],
                                        show_color_wheel=True)
        fig.savefig(os.path.join(save_folder, f"{i}.png"), bbox_inches='tight')
        plt.close("all")

        fig = plt.figure(figsize=(10, 10))
        ax = plt.subplot(221)
        ax.imshow(npimages[i])
        ax = plt.subplot(222)
        ax.imshow(npimages_next[i])
        ax = plt.subplot(223, projection='3d')
        rots = R.from_matrix(mean_numpy[i]).as_rotvec()  # .as_euler('zxy', degrees=False)
        ax.scatter3D(rots[:, 0], rots[:, 1], rots[:, 2], s=[30] * len(rots))
        ax = plt.subplot(224, projection='3d')
        rots = R.from_matrix(mean_next[i]).as_rotvec()  # .as_euler('zxy', degrees=False)
        ax.scatter3D(rots[:, 0], rots[:, 1], rots[:, 2], s=[30] * len(rots), c="r", alpha=0.1)
        rots = R.from_matrix(mean_rot[i]).as_rotvec()  # .as_euler('zxy', degrees=False)
        ax.scatter3D(rots[:, 0], rots[:, 1], rots[:, 2], s=[30] * len(rots), marker="*")
        fig.savefig(os.path.join(save_folder, f"rotvec_{i}.png"), bbox_inches='tight')
        plt.close("all")

        fig = plt.figure(figsize=(5, 5))
        ax = plt.subplot(111)
        add_image_to_ax(npimages_next[i], ax=ax)
        fig.savefig(os.path.join(save_folder, f"image_alone_{i}.png"), bbox_inches='tight')
        plt.close("all")
        fig = plt.figure(figsize=(5, 5))
        ax = plt.subplot(111, projection='3d')
        ax.scatter3D(rots[:, 0], rots[:, 1], rots[:, 2], s=[30] * len(rots), c="r", alpha=0.5)
        N = 200
        stride = 1
        u = np.linspace(0, 2 * np.pi, N)
        v = np.linspace(0, np.pi, N)
        radius = np.sqrt(np.pi)
        x = np.outer(np.cos(u) * radius, np.sin(v) * radius)
        y = np.outer(np.sin(u) * radius, np.sin(v) * radius)
        z = np.outer(np.ones(np.size(u)) * radius, np.cos(v) * radius)
        ax.plot_surface(x, y, z, linewidth=0.0, cstride=stride, alpha=0.05, color="k", rstride=stride)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        fig.savefig(os.path.join(save_folder, f"rotvec_alone_{i}.png"), bbox_inches='tight')

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
if run is not None:
    run.stop()
