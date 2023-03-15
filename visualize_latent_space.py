import argparse
import pickle
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from utils.dataset_utils import get_dataset
from utils.nn_utils import get_rotated_mean, make_rotation_matrix, hitRate_generic, so2_rotate_subspaces
from utils.plotting_utils import plot_extra_dims, plot_images_distributions, \
    plot_mixture_neurreps, add_image_to_ax, add_distribution_to_ax_torus, save_embeddings_on_circle, yiq_embedding, \
    plot_embeddings_eval_torus, plot_projected_embeddings_pca
from utils.plotting_utils_so3 import visualize_so3_probabilities
from utils.torch_utils import torch_data_to_numpy
from utils.disentanglement_metric import dlsbd_metric_mixture
from datasets.equiv_dset import EquivDataset

# region PARSE ARGUMENTS
parser = argparse.ArgumentParser()
parser.add_argument('--save-folder', type=str, default='checkpoints', help='Path to saved model')
parser.add_argument('--dataset', type=str, default='dataset', help='Dataset')
parser.add_argument('--dataset_name', nargs="+", type=str, default=['4'], help='Dataset name')
args_eval = parser.parse_args()
# endregion

# region PATHS
model_dir = os.path.join(".", "saved_models", args_eval.save_folder)
model_file = os.path.join(model_dir, 'model.pt')
decoder_file = os.path.join(model_dir, 'decoder.pt')
meta_file = os.path.join(model_dir, 'metadata.pkl')

# endregion

# region LOAD METADATA
args = pickle.load(open(meta_file, 'rb'))['args']
print(args)
save_folder = os.path.join(".", "visualizations", args.model_name)
os.makedirs(save_folder, exist_ok=True)
device = torch.device("cuda:" + args.gpu if torch.cuda.is_available() else "cpu")
# torch.manual_seed(42)
torch.cuda.empty_cache()
# endregion

# region SETUP NEPTUNE
if args.neptune_user != "":
    from utils.neptune_utils import reload_neptune_run

    neptune_id_file = os.path.join(model_dir, 'neptune.txt')
    run = reload_neptune_run(args.neptune_user, "non-free", neptune_id_file)
else:
    run = None
# endregion

# region LOAD DATASET
dset, eval_dset = get_dataset(args.data_dir, args.dataset, args.dataset_name)
train_loader = torch.utils.data.DataLoader(dset, batch_size=100, shuffle=True)
# endregion

# region GET MODEL
print("Loading model from: ", model_file)
device = 'cpu'
model = torch.load(model_file).to(device)
if args.autoencoder != 'None':
    decoder = torch.load(decoder_file).to(device)
    decoder.eval()
else:
    decoder = None
model.eval()
# endregion

# region GET IMAGES
print("Getting images")
img, img_next, action, n_stabilizers = next(iter(train_loader))
img_shape = np.array(img.shape[1:])
print("Flat images shape", eval_dset.flat_images.shape)

flat_images_tensor = torch.Tensor(eval_dset.flat_images).to(device)  # transform to torch tensor
eval_tensor_dset = torch.utils.data.TensorDataset(flat_images_tensor)  # create your datset
eval_dataloader = torch.utils.data.DataLoader(eval_tensor_dset, batch_size=args.batch_size)

# Get the numpy array versions of the images
print("Getting numpy arrays")
npimages = torch_data_to_numpy(img)
npimages_next = torch_data_to_numpy(img_next)
npimages_eval = eval_dset.flat_images_numpy
# endregion

# region POSTERIOR PARAMETERS
print("Computing posterior parameters")
# Calculate the parameters obtained by the models
mean, logvar, extra = model(img.to(device))
extra = extra.detach().cpu().numpy()
mean_next, logvar_next, extra_next = model(img_next.to(device))
extra_next = extra_next.detach().cpu().numpy()

# Obtain the values as numpy arrays

# endregion

mean_numpy = mean.detach().cpu().numpy()
mean_next = mean_next.detach().cpu().numpy()
std = np.exp(logvar.detach().cpu().numpy() / 2.) / 10
print("Distribution of the standard deviations", np.unique(std * 10))
std_next = np.exp(logvar_next.detach().cpu().numpy() / 2.) / 10

# Plotting for SO(2) and its combinations
if args.latent_dim == 2 or args.latent_dim == 4:
    # region GET EMBEDDINGS
    mean_eval = []
    logvar_eval = []
    extra_eval = []
    for num_batch, batch in enumerate(eval_dataloader):
        print("Encoding batch", num_batch)
        mean_eval_, logvar_eval_, extra_eval_ = model(batch[0])
        mean_eval.append(mean_eval_)
        logvar_eval.append(logvar_eval_)
        extra_eval.append(extra_eval_)
    mean_eval = torch.cat(mean_eval, dim=0)
    logvar_eval = torch.cat(logvar_eval, dim=0)
    extra_eval = torch.cat(extra_eval, dim=0)

    if not (args.variablescale):
        logvar = -4.6 * torch.ones(logvar.shape).to(logvar.device)
        logvar_eval = -4.6 * torch.ones(logvar_eval.shape).to(logvar_eval.device)
    std_eval = np.exp(logvar_eval.detach().cpu().numpy() / 2.) / 10

    action = action.squeeze(1)
    mean_rot = get_rotated_mean(mean, action, args.latent_dim)
    action = action.detach().cpu().numpy()
    # endregion

    # region PLOT STD DISTRIBUTION
    if run is not None:
        fig, ax = plt.subplots(1, 1)
        ax.hist(std_eval.flatten())
        ax.set_xlabel("Standard deviation")
        ax.set_ylabel("Frequency")
        run["eval_std_hist"].upload(plt.gcf())
    # endregion

    # region PLOT IDENTITY EMBEDDINGS
    if (args.latent_dim != 4) and (args.extra_dim > 0):
        # assert if n_stabilizers is string
        n_stabilizers = n_stabilizers.squeeze(1).detach().cpu().numpy()

        fig, ax = plot_projected_embeddings_pca(extra, n_stabilizers)
        if fig:
            fig.savefig(os.path.join(save_folder, 'invariant.png'))
            if run is not None:
                run["invariant"].upload(fig)
        try:
            extra_repeated = np.expand_dims(extra, axis=1)
            n_value = mean.shape[1]
            print("Unique extra repeated shape", extra_repeated.shape)

            extra_repeated = np.tile(extra_repeated, (1, n_value, 1))
            print("Unique extra repeated shape", extra_repeated.shape)
            extra_repeated = extra_repeated.reshape((-1, extra_repeated.shape[-1]))
            mean_reshaped = mean.detach().cpu().numpy().reshape((-1, mean.shape[-1]))
            print("Mean shape", mean_reshaped.shape, extra_repeated.shape)
            concatenated_embeddings = np.concatenate([mean_reshaped, extra_repeated], axis=-1)
            print("Concatenated shape", concatenated_embeddings.shape)
            fig, ax = plot_projected_embeddings_pca(concatenated_embeddings)
            if fig:
                fig.savefig(os.path.join(save_folder, 'projected.png'))
                if run is not None:
                    run["projected"].upload(fig)
        except:
            print("Could not plot PCA")

    # endregion

    # region PLOT ROTATED EMBEDDINGS
    for j in range(10):
        print(f"Plotting example {j}")

        fig, axes = plot_images_distributions(mean=mean_numpy[j], std=std[j], mean_next=mean_next[j],
                                              std_next=std_next[j],
                                              image=npimages[j], image_next=npimages_next[j],
                                              expected_mean=mean_rot[j], n=args.num)
        plt.savefig(os.path.join(save_folder, f"image_pair_{j}.png"), bbox_inches='tight')
        if run is not None:
            run[f"image_pair_{j}"].upload(plt.gcf())
        plot_mixture_neurreps(mean_numpy[j])

        # For submission
        # add_image_to_ax(npimages[i])
        # if run is not None:
        #     run[f"image_{i}"].upload(plt.gcf())
        # plt.savefig(os.path.join(save_folder, f"test_image_{i}.png"), bbox_inches='tight')
        # if run is not None:
        #     run[f"test_mixture_{i}"].upload(plt.gcf())
        # plt.savefig(os.path.join(save_folder, f"test_mixture_{i}.png"), bbox_inches='tight')

        # fig.savefig(os.path.join(save_folder, f"test_{i}.png"), bbox_inches='tight')
        plt.close("all")
    # endregion

    # region PLOTS ON THE CIRCLE
    if args.latent_dim != 4:
        save_embeddings_on_circle(mean_eval.detach().cpu().numpy(), std_eval, eval_dset.flat_stabs, save_folder,
                                  args.dataset_name[0],
                                  increasing_radius=True)
        if run is not None:
            run["embeddings_on_circle"].upload(plt.gcf())
    # endregion

    # region PLOT ON THE TORUS
    if args.latent_dim == 4:
        print("EVAL ACTIONS SHAPE", eval_dset.flat_lbls.shape)
        colors = yiq_embedding(eval_dset.flat_lbls[:, 0], eval_dset.flat_lbls[:, 1])
        fig, ax = plot_embeddings_eval_torus(mean_eval.detach().numpy(), colors)
        fig.savefig(os.path.join(save_folder, f"torus_eval_embedings.png"), bbox_inches='tight')
        plt.close("all")
    # endregion

    # region GET UNIQUE EMBEDDINGS
    unique_images = []
    for unique in np.unique(dset.stabs, axis=0):
        if args.latent_dim == 4:
            # If latent dim is 4 the stabilizer is a 2D vector
            unique_images.append(dset.data[np.product(dset.stabs == unique, axis=-1)][0][0])
        else:
            unique_images.append(dset.data[dset.stabs == unique][0][0])

    unique_images = torch.tensor(np.array(unique_images), dtype=img.dtype).to(device)
    if len(unique_images.shape) == 3:
        unique_images = unique_images.unsqueeze(0)
    print("Unique images shape", unique_images.shape)
    unique_mean, unique_logvar, unique_extra = model(unique_images)
    unique_std = np.exp(unique_logvar.detach().cpu().numpy() / 2.) / 10
    print(unique_mean.shape, unique_extra.shape)
    # endregion

    # region PLOT SUBMISSION
    for num_unique in range(len(unique_images)):
        add_image_to_ax(np.transpose(unique_images[num_unique].detach().cpu().numpy(), (1, 2, 0)))
        plt.savefig(os.path.join(save_folder, f"test_image_{num_unique}.png"), bbox_inches='tight')
        if run is not None:
            run[f"image_{num_unique}"].upload(plt.gcf())
        # Plot mixture distribution
        if args.latent_dim == 4:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax = add_distribution_to_ax_torus(unique_mean[num_unique].detach().cpu().numpy(), unique_std[num_unique],
                                              ax, n=args.num, color="r", scatter=True)
            ax.set_xlim([-np.pi, np.pi])
            ax.set_ylim([-np.pi, np.pi])
            ax.set_xticks([])
            ax.set_yticks([])

            plt.savefig(os.path.join(save_folder, f"test_mixture_{num_unique}.png"), bbox_inches='tight')
        else:
            plot_mixture_neurreps(unique_mean[num_unique].detach().cpu().numpy())
        if run is not None:
            run[f"test_mixture_{num_unique}"].upload(plt.gcf())
        plt.savefig(os.path.join(save_folder, f"test_mixture_{num_unique}.png"), bbox_inches='tight')
    # endregion

    # region PLOT RECONSTRUCTIONS
    if args.autoencoder != "None":
        if args.autoencoder == "ae_single" or args.autoencoder == "vae":
            if args.autoencoder == "vae":
                extra_loc = unique_extra[:, -2 * args.extra_dim: args.extra_dim]
                x_rec = decoder(torch.cat([unique_mean[:, 0], extra_loc], dim=-1))
            else:
                x_rec = decoder(torch.cat([unique_mean[:, 0], unique_extra], dim=-1))
            x_rec = x_rec.permute((0, 2, 3, 1)).detach().cpu().numpy()
            for j in range(len(x_rec)):
                add_image_to_ax(1 / (1 + np.exp(-x_rec[j])))
                if run is not None:
                    run[f"reconstruction_{j}"].upload(plt.gcf())
                plt.savefig(os.path.join(save_folder, f"reconstruction_{j}.png"), bbox_inches='tight')
                add_image_to_ax(unique_images[j].permute((1, 2, 0)).detach().cpu().numpy())
                plt.savefig(os.path.join(save_folder, f"input_image_{j}.png"), bbox_inches='tight')
        else:
            x_rec = decoder(torch.cat([unique_mean.view((unique_mean.shape[0], -1)), unique_extra], dim=-1))
            x_rec = x_rec.permute((0, 2, 3, 1)).detach().cpu().numpy()
            for j in range(len(x_rec)):
                add_image_to_ax(1 / (1 + np.exp(-x_rec[j])))
                if run is not None:
                    run[f"reconstruction_{j}"].upload(plt.gcf())
                plt.savefig(os.path.join(save_folder, f"reconstruction_{j}.png"), bbox_inches='tight')
                add_image_to_ax(unique_images[j].permute((1, 2, 0)).detach().cpu().numpy())
                plt.savefig(os.path.join(save_folder, f"input_image_{j}.png"), bbox_inches='tight')

            # boolean_selection = (flat_stabilizers == unique)
            # fig, _ = plot_images_multi_reconstructions(npimages_eval[boolean_selection][:5],
            #                                            reconstructions_np[boolean_selection][:5])
            # fig.savefig(os.path.join(save_folder, f"{unique}_reconstructions.png"), bbox_inches='tight')
    # endregion

    # region LATENT TRAVERSAL
    if args.latent_dim != 4:
        num_points_traversal = 15
        num_objects = unique_mean.shape[0]
        if args.autoencoder != "None":
            angles_traversal = np.linspace(0, 2 * np.pi, num_points_traversal, endpoint=False)
            angles_traversal = torch.tensor(angles_traversal).float()

            if args.latent_dim == 2:
                # Rotate the embeddings
                rotated_embeddings = []
                print("Angles traversal shape", angles_traversal.shape)
                rot = make_rotation_matrix(angles_traversal).unsqueeze(0)
                print("Rotation matrices shape", rot.shape)
                rotated_embeddings = (rot @ unique_mean.unsqueeze(-1).unsqueeze(1).float()).squeeze(-1)
                print("Rotated embeddings shape", rotated_embeddings.shape)
                rotated_embeddings = rotated_embeddings.view(
                    (-1, rotated_embeddings.shape[-1] * rotated_embeddings.shape[-2]))
                print("Rotated embeddings shape", rotated_embeddings.shape)
                # Rotated embeddings shape (num_points_traversal, num_unique, N, latent_dim)

            # Unique extra shape (num_unique, extra_dim)
            if args.extra_dim > 0:
                if args.autoencoder == "vae":
                    extra_repeated = unique_extra[:, -2 * args.extra_dim: args.extra_dim].unsqueeze(0)
                else:
                    extra_repeated = unique_extra.unsqueeze(0)
                print("Unique extra repeated shape", extra_repeated.shape)
                if num_objects > 1:
                    extra_repeated = extra_repeated.repeat((1, num_points_traversal, 1),
                                                           0)
                else:
                    extra_repeated = extra_repeated.repeat((num_points_traversal, 1), 0)
                extra_repeated = extra_repeated.view((-1, extra_repeated.shape[-1]))
                print("Unique extra repeated shape", extra_repeated.shape)
                # mean_extra = mean_extra.repeat(num_points_traversal, 0)
                if args.autoencoder == "ae_single" or args.autoencoder == "vae":
                    rotated_embeddings = rotated_embeddings.view(num_points_traversal, num_objects, args.num,
                                                                 args.latent_dim)
                    rotated_embeddings = rotated_embeddings[:, :, 0, :]
                    print("Rotated embeddings shape", rotated_embeddings.shape)
                    extra_repeated = extra_repeated.view(num_points_traversal, num_objects, extra_repeated.shape[-1])
                    print("Extra repeated shape", extra_repeated.shape)
                    z = torch.cat([rotated_embeddings, extra_repeated], dim=-1).float()
                    print("Z shape", z.shape)
                    z = z.view((-1, z.shape[-1]))
                else:
                    z = torch.cat([rotated_embeddings, extra_repeated], dim=-1).float()

            else:
                if args.autoencoder == "ae_single":
                    z = rotated_embeddings.reshape(num_points_traversal * num_objects, args.num, args.latent_dim)[:, 0,
                        :]
                else:
                    z = rotated_embeddings
            print("Z shape", z.shape)
            x_rec = decoder(z)
            x_rec = x_rec.permute((0, 2, 3, 1)).detach().cpu().numpy()
            x_rec = x_rec.reshape((num_points_traversal, num_objects, *x_rec.shape[1:]))
            fig, axes = plt.subplots(num_objects, num_points_traversal, figsize=(num_points_traversal, 1 * num_objects))
            for i in range(num_objects):
                for j in range(num_points_traversal):
                    if axes.ndim == 1:
                        axes[j].imshow(x_rec[j, i])
                        axes[j].axis("off")
                    else:
                        axes[i, j].imshow(x_rec[j, i])
                        axes[i, j].axis('off')
            fig.savefig(os.path.join(save_folder, f"traversal.png"), bbox_inches='tight')
            if run is not None:
                run["traversal"].upload(plt.gcf())
    # endregion

    # region ESTIMATE DLSBD METRIC
    if args.latent_dim == 4:
        reshaped_eval_actions = eval_dset.flat_lbls.reshape((eval_dset.num_objects, -1, 2))
        reshaped_stabs = eval_dset.flat_stabs.reshape((eval_dset.num_objects, -1, 2))
    else:
        reshaped_eval_actions = eval_dset.flat_lbls.reshape((eval_dset.num_objects, -1))
        reshaped_stabs = eval_dset.flat_stabs.reshape((eval_dset.num_objects, -1))

    reshaped_mean_eval = mean_eval.reshape(
        (eval_dset.num_objects, -1, args.num, args.latent_dim)).detach().cpu().numpy()

    print("Eval actions", type(reshaped_eval_actions), "Mean type", type(reshaped_mean_eval), "Stabs type",
          type(reshaped_stabs))
    print("Eval actions shape", reshaped_eval_actions.shape, "Mean shape", reshaped_mean_eval.shape, "Stabs shape",
          reshaped_stabs.shape)
    dlsbd = dlsbd_metric_mixture(reshaped_mean_eval, reshaped_eval_actions, reshaped_stabs, distance_function="chamfer")
    print("DLSBD Metric", dlsbd)
    model_path = os.path.join(args.checkpoints_dir, args.model_name)
    np.save(f'{model_path}/dlsbd.npy', [dlsbd])
    if run is not None:
        run["metrics/dlsbd"].log(dlsbd)
    # endregion

    # region ESTIMATE HIT-RATE

    print(f"Loading dataset {args.dataset} with dataset name {args.dataset_name}")
    dset_val = EquivDataset(f'{args.data_dir}/{args.dataset}/',
                            list_dataset_names=[dataset_name + "_val" for dataset_name in args.dataset_name],
                            max_data_per_dataset=-1)
    # Note that the batch size is fixed to 20
    eval_batch_size = 20
    val_loader = torch.utils.data.DataLoader(dset_val, batch_size=eval_batch_size, shuffle=True)
    print("# test set:", len(dset_val))
    mu_hit_rate = 0
    total_batches = len(val_loader)
    for batch_idx, (image, img_next, action) in enumerate(val_loader):
        batch_size = image.shape[0]
        image = image.to(device)
        img_next = img_next.to(device)
        action = action.to(device)

        z_mean, z_logvar, extra = model(image)
        z_mean_next, z_logvar_next, extra_next = model(img_next)
        if args.latent_dim == 2:
            action = torch.flatten(action)
            rot = make_rotation_matrix(action)
            z_mean_pred = (rot @ z_mean.unsqueeze(-1)).squeeze(-1)  # Beware the detach!!!
        elif args.latent_dim == 3:
            z_mean_pred = action @ z_mean
        elif args.latent_dim > 3 and args.latent_dim % 2 == 0:
            action = action.squeeze(1)
            z_mean_pred = so2_rotate_subspaces(z_mean, action, detach=False)
        else:
            raise ValueError(f"Rotation not defined for latent dimension {args.latent_dim} ")

        if args.latent_dim == 3:
            chamfer_matrix = \
                ((z_mean_pred.unsqueeze(1).unsqueeze(1) - z_mean_next.unsqueeze(2).unsqueeze(0)) ** 2).sum(-1).sum(
                    -1).min(
                    dim=-1)[
                    0].sum(dim=-1)
        else:
            chamfer_matrix = \
                ((z_mean_pred.unsqueeze(1).unsqueeze(1) - z_mean_next.unsqueeze(2).unsqueeze(0)) ** 2).sum(-1).min(
                    dim=-1)[
                    0].sum(dim=-1)
        if args.autoencoder == "vae":
            loc_extra = extra[:, -2 * args.extra_dim: -args.extra_dim]
            loc_extra_next = extra_next[:, -2 * args.extra_dim: -args.extra_dim]
            extra_matrix = ((loc_extra.unsqueeze(0) - loc_extra_next.unsqueeze(1)) ** 2).sum(-1)
        else:
            extra_matrix = ((extra.unsqueeze(0) - extra_next.unsqueeze(1)) ** 2).sum(-1)
        hitrate = hitRate_generic(chamfer_matrix + extra_matrix, image.shape[0])
        mu_hit_rate += hitrate.item()
    mu_hit_rate /= total_batches
    if run is not None:
        run["metrics/hitrate"].log(mu_hit_rate)
    print("Hit rate", mu_hit_rate)
    np.save(f'{model_path}/errors_hitrate20.npy', [mu_hit_rate])
    # endregion

# region PLOTTING SO(3)
elif args.latent_dim == 3:

    unique_images = []
    num_unique_examples = 5
    num_unique_orbits = len(np.unique(dset.stabs))
    for unique in np.unique(dset.stabs):
        unique_images.append(dset.data[dset.stabs == unique][:num_unique_examples, 0])

    unique_images = torch.tensor(np.array(unique_images).reshape(-1, *unique_images[0].shape[-3:]), dtype=img.dtype).to(
        device)
    print(unique_images.shape)
    unique_mean, unique_logvar, unique_extra = model(unique_images)
    unique_images = unique_images.reshape((num_unique_orbits, num_unique_examples, *unique_images.shape[-3:]))
    # region PLOT RECONSTRUCTIONS
    if args.autoencoder != "None":
        if args.autoencoder == "ae_single":
            unique_mean = unique_mean[:, 0]
            x_rec = decoder(torch.cat([unique_mean.view((unique_mean.shape[0], -1)), unique_extra], dim=-1))
        else:
            x_rec = decoder(torch.cat([unique_mean.view((unique_mean.shape[0], -1)), unique_extra], dim=-1))
        x_rec = x_rec.permute((0, 2, 3, 1)).detach().cpu().numpy()
        x_rec = x_rec.reshape((num_unique_orbits, num_unique_examples, *x_rec.shape[-3:]))
        for j in range(len(x_rec)):
            fig, axes = plt.subplots(1, num_unique_examples)
            fig2, axes2 = plt.subplots(1, num_unique_examples)
            for k, rec in enumerate(x_rec[j]):
                add_image_to_ax(1 / (1 + np.exp(-rec)), axes[k])
                add_image_to_ax(unique_images[j, k].permute((1, 2, 0)).detach().cpu().numpy(), axes2[k])
            fig.savefig(os.path.join(save_folder, f"reconstruction_{j}.png"), bbox_inches='tight')
            if run is not None:
                run["reconstruction" + str(j)].upload(fig)
            if run is not None:
                run["input_image" + str(j)].upload(fig2)
            fig2.savefig(os.path.join(save_folder, f"input_image_{j}.png"), bbox_inches='tight')

    for j in range(10):
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
        ax.imshow(npimages[j])
        ax.set_xticks([])
        ax.set_yticks([])
        ax = plt.subplot(222)
        ax.imshow(npimages_next[j])
        ax.set_xticks([])
        ax.set_yticks([])
        ax = plt.subplot(223, projection='mollweide')

        _ = visualize_so3_probabilities(mean_numpy[j], 0.05 * np.ones(len(mean_numpy[j])), ax=ax, fig=fig,
                                        show_color_wheel=True)
        ax = plt.subplot(224, projection='mollweide')
        _ = visualize_so3_probabilities(mean_next[j], 0.05 * np.ones(len(mean_next[j])), ax=ax, fig=fig,
                                        rotations_gt=mean_rot[j],
                                        show_color_wheel=True)
        fig.savefig(os.path.join(save_folder, f"{j}.png"), bbox_inches='tight')
        if run is not None:
            run["probabilities_" + str(j)].upload(plt.gcf())
        plt.close("all")

        fig = plt.figure(figsize=(10, 10))
        ax = plt.subplot(221)
        ax.imshow(npimages[j])
        ax = plt.subplot(222)
        ax.imshow(npimages_next[j])
        ax = plt.subplot(223, projection='3d')
        rots = R.from_matrix(mean_numpy[j]).as_rotvec()  # .as_euler('zxy', degrees=False)
        ax.scatter3D(rots[:, 0], rots[:, 1], rots[:, 2], s=[30] * len(rots))
        ax = plt.subplot(224, projection='3d')
        rots = R.from_matrix(mean_next[j]).as_rotvec()  # .as_euler('zxy', degrees=False)
        ax.scatter3D(rots[:, 0], rots[:, 1], rots[:, 2], s=[30] * len(rots), c="r", alpha=0.1)
        rots = R.from_matrix(mean_rot[j]).as_rotvec()  # .as_euler('zxy', degrees=False)
        ax.scatter3D(rots[:, 0], rots[:, 1], rots[:, 2], s=[30] * len(rots), marker="*")
        fig.savefig(os.path.join(save_folder, f"rotvec_{j}.png"), bbox_inches='tight')
        if run is not None:
            run["rotvec" + str(j)].upload(plt.gcf())
        plt.close("all")

        fig = plt.figure(figsize=(5, 5))
        ax = plt.subplot(111)
        add_image_to_ax(npimages_next[j], ax=ax)
        fig.savefig(os.path.join(save_folder, f"image_alone_{j}.png"), bbox_inches='tight')
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
        if run is not None:
            run["rotvec_alone" + str(j)].upload(plt.gcf())
        fig.savefig(os.path.join(save_folder, f"rotvec_alone_{j}.png"), bbox_inches='tight')

# endregion
if run is not None:
    run["sys/failed"] = False
    run.stop()
