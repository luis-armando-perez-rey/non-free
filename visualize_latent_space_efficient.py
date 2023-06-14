import argparse
import pickle
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from utils.dataset_utils import get_dataset, get_data_from_dataloader, get_loading_parameters
from utils.nn_utils import get_rotated_mean, make_rotation_matrix, hitRate_generic, so2_rotate_subspaces
from utils.plotting_utils import plot_extra_dims, plot_images_distributions, \
    plot_mixture_neurreps, add_image_to_ax, add_distribution_to_ax_torus, save_embeddings_on_circle, yiq_embedding, \
    plot_embeddings_eval_torus, plot_projected_embeddings_pca, plot_std_distribution, plot_cylinder, \
    plot_image_mixture_rec_all, plot_clusters
from utils.plotting_utils_so3 import visualize_so3_probabilities
from utils.torch_utils import torch_data_to_numpy
from utils.disentanglement_metric import dlsbd_metric_mixture, dlsbd_metric_mixture_monte
from utils.model_utils import reload_model, get_embeddings, get_reconstructions, sigmoid, get_n_clusters_noise
from datasets.equiv_dset import EquivDataset
from dataset_generation.modelnet_efficient import ModelNetUniqueDataset, ModelNetDataset

# region PARSE ARGUMENTS
parser = argparse.ArgumentParser()
parser.add_argument('--save-folder', type=str, default='checkpoints', help='Path to saved model')
parser.add_argument('--dataset', type=str, default='dataset', help='Dataset')
parser.add_argument('--dataset_name', nargs="+", type=str, default=['4'], help='Dataset name')
args_eval = parser.parse_args()
# endregion

# region LOAD METADATA
model_dir = os.path.join(".", "saved_models", args_eval.save_folder)
meta_file = os.path.join(model_dir, 'metadata.pkl')
args = pickle.load(open(meta_file, 'rb'))['args']
print("Arguments", args)
save_folder = os.path.join(".", "visualizations", args.model_name)
os.makedirs(save_folder, exist_ok=True)
if args.gpu != "":
    device = "cpu"
else:
    device = torch.device("cuda:" + args.gpu if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
# torch.manual_seed(42)
torch.cuda.empty_cache()
# endregion

# region SETUP NEPTUNE
neptune_id_file = os.path.join(model_dir, 'neptune.txt')
if args.neptune_user != "":
    from utils.neptune_utils import reload_neptune_run

    run = reload_neptune_run(args.neptune_user, "non-free", neptune_id_file)
else:
    run = None
# endregion

# region LOAD DATASET
dset, eval_dset = get_dataset(args.data_dir, args.dataset, args.dataset_name)
train_loader = torch.utils.data.DataLoader(dset, batch_size=100, shuffle=True, num_workers=4, pin_memory=True)

# endregion

# region GET MODEL
model, decoder = reload_model(model_dir, args.autoencoder, device=device)
model_path = os.path.join(args.checkpoints_dir, args.model_name)
# endregion

# region GET IMAGES
print("Getting images")
output = next(iter(train_loader))

n_data_elements = len(output)
if n_data_elements == 6:
    img, img_next, action, n_stabilizers, orbits, object_types = output
    identifiers = orbits.detach().cpu().numpy()
elif n_data_elements == 5:
    img, img_next, action, n_stabilizers, orbits = output
    identifiers = orbits.detach().cpu().numpy()
else:
    img, img_next, action, n_stabilizers = output
    identifiers = n_stabilizers.squeeze(1).detach().cpu().numpy()
print("Identifiers shape", identifiers.shape)

img_shape = np.array(img.shape[1:])

if not(args.dataset.startswith("modelnet_efficient") or args.dataset.startswith("shrec21shape")):
    flat_images_tensor = torch.Tensor(eval_dset.flat_images).to(device)  # transform to torch tensor
    eval_tensor_dset = torch.utils.data.TensorDataset(flat_images_tensor)  # create your datset
    print("Flat images shape", eval_dset.flat_images.shape)
else:
    eval_tensor_dset = eval_dset
eval_dataloader = torch.utils.data.DataLoader(eval_tensor_dset, batch_size=args.batch_size, num_workers=4,
                                              pin_memory=True)

# Get the numpy array versions of the images
print("Getting numpy arrays")
npimages = torch_data_to_numpy(img)
npimages_next = torch_data_to_numpy(img_next)
# endregion

# region POSTERIOR PARAMETERS
print("Computing posterior parameters")
# Calculate the parameters obtained by the models
mean, logvar, extra = model(img.to(device).float())
extra = extra.detach().cpu().numpy()
mean_next, logvar_next, extra_next = model(img_next.to(device))
extra_next = extra_next.detach().cpu().numpy()
if not (args.variablescale):
    logvar = -4.6 * torch.ones_like(logvar).to(logvar.device)

# Obtain the values as numpy arrays

# endregion

mean_numpy = mean.detach().cpu().numpy()
mean_next = mean_next.detach().cpu().numpy()
std = np.exp(logvar.detach().cpu().numpy() / 2.) / 10
std_next = np.exp(logvar_next.detach().cpu().numpy() / 2.) / 10

# Plotting for SO(2) and its combinations
if args.latent_dim == 2 or args.latent_dim == 4:

    mean_eval, logvar_eval, std_eval, extra_eval = get_embeddings(eval_dataloader, model, args.variablescale, device)

    action = action.squeeze(1)
    mean_rot = get_rotated_mean(mean, action, args.latent_dim)
    action = action.detach().cpu().numpy()
    # endregion

    # region ESTIMATE DLSBD METRIC
    if args.latent_dim == 4:
        reshaped_eval_actions = eval_dset.flat_lbls.reshape((eval_dset.num_objects, eval_dset.num_views, 2))
        reshaped_stabs = eval_dset.flat_stabs.reshape((eval_dset.num_objects, eval_dset.num_views, 2))
        eval_stabs = eval_dset.flat_stabs
    else:
        if args.dataset.startswith("modelnet_efficient") or args.dataset.startswith("shrec21shape"):
            eval_actions = get_data_from_dataloader(eval_dataloader, 1).numpy()
            eval_stabs = get_data_from_dataloader(eval_dataloader, 2).numpy()
            eval_orbits = get_data_from_dataloader(eval_dataloader, 3).numpy()
            eval_object_types = get_data_from_dataloader(eval_dataloader, 4).numpy()
            reshaped_eval_actions = eval_actions.reshape((eval_dset.num_objects, eval_dset.num_views))
            reshaped_stabs = eval_stabs.reshape((eval_dset.num_objects, eval_dset.num_views))
        else:
            eval_stabs = eval_dset.flat_stabs
            reshaped_eval_actions = eval_dset.flat_lbls.reshape((eval_dset.num_objects, eval_dset.num_views))
            reshaped_stabs = eval_dset.flat_stabs.reshape((eval_dset.num_objects, eval_dset.num_views))

    # Dataset with shape (num_objects, num_views, N, latent_dim)
    reshaped_mean_eval = mean_eval.reshape(
        (eval_dset.num_objects, eval_dset.num_views, args.num, args.latent_dim)).detach().cpu().numpy()

    dlsbd = dlsbd_metric_mixture(reshaped_mean_eval, reshaped_eval_actions, reshaped_stabs, distance_function="chamfer")
    dlsbd_monte = dlsbd_metric_mixture_monte(reshaped_mean_eval, reshaped_eval_actions, reshaped_stabs,
                                             distance_function="chamfer")
    print("DLSBD Metric", dlsbd)
    print("DLSBD Metric Monte", dlsbd_monte)

    np.save(f'{model_path}/dlsbd.npy', [dlsbd])
    np.save(f'{model_path}/dlsbd_monte.npy', [dlsbd_monte])
    if run is not None:
        run["metrics/dlsbd"].log(dlsbd)
        run["metrics/dlsbd_monte"].log(dlsbd)
    # endregion

    # region PLOT STD DISTRIBUTION
    if run is not None:
        fig, ax = plot_std_distribution(std, ax=None)
        run["plots/eval_std_hist"].upload(fig)
        plt.close(fig)
    # endregion

    # region PLOT IDENTITY EMBEDDINGS
    if (args.latent_dim != 4) and (args.extra_dim > 0):
        fig, ax = plot_projected_embeddings_pca(extra, identifiers)
        fig.savefig(os.path.join(save_folder, 'invariant.png'))
        if run is not None:
            run["plots/invariant"].upload(fig)
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
                    run["plots/projected"].upload(fig)
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
        plt.close("all")
    # endregion

    # region PLOTS ON THE CIRCLE
    # TODO: Remove plot on circle since it needs to be updated for this dataset
    # if args.latent_dim != 4:
    #     if n_data_elements == 6:
    #         print("EVAL ACTIONS SHAPE", eval_dset.factors[1].shape)
    #         print("Mean eval shape", mean_eval.shape)
    #         figures = save_embeddings_on_circle(mean_eval.detach().cpu().numpy(), std_eval, eval_object_types,
    #                                             save_folder,
    #                                             args.dataset_name[0],
    #                                             increasing_radius=True)
    #     else:
    #         figures = save_embeddings_on_circle(mean_eval.detach().cpu().numpy(), std_eval, eval_stabs,
    #                                             save_folder,
    #                                             args.dataset_name[0],
    #                                             increasing_radius=True)
    #     if run is not None:
    #         for i, fig in enumerate(figures):
    #             run[f"embeddings_on_circle_{i}"].upload(fig)
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

    train_data_parameters = get_loading_parameters(args.data_dir, args.dataset, args.dataset_name)[0]
    train_data_parameters.pop("shuffle_available_views")
    # Select by the object orbits
    dset_unique = ModelNetUniqueDataset(**train_data_parameters, index_unique_factors=-2)
    dset_unique_loader = torch.utils.data.DataLoader(dset_unique, batch_size=10, shuffle=False, num_workers=4,
                                                     pin_memory=True)

    unique_mean, unique_logvar, unique_std, unique_extra = get_embeddings(dset_unique_loader, model, args.variablescale, device)
    unique_std = np.exp(unique_logvar.detach().cpu().numpy() / 2.) / 10
    print(unique_mean.shape, unique_extra.shape)
    # endregion

    # region PLOT CLUSTER DISTRIBUTION
    n_clusters = []
    n_noise = []
    for mean in unique_mean:
        clusters, noise = get_n_clusters_noise(mean.detach().cpu().numpy())
        n_clusters.append(clusters)
        n_noise.append(noise)

    fig, ax = plot_clusters(n_clusters, args.num)
    if run is not None:
        run["plots/n_clusters"].upload(fig)
    # if n_data_elements == 5 or n_data_elements == 6:
    #     for num_object_type, unique_object_type in enumerate(np.unique(unique_object_types)):
    #         fig, ax = plot_clusters(n_clusters[unique_object_types == unique_object_type], args.num, ax,
    #                                 label=str(unique_object_type))
    #     if run is not None:
    #         run["plots/n_clusters"].upload(fig)

    # endregion

    # region PLOT CYLINDER
    fig, ax = plot_cylinder(mean_eval.detach().cpu().numpy(), eval_dset.num_views, top_view=False)
    if run is not None:
        run["plots/embeddings_cylinder"].upload(fig)

    fig, ax = plot_cylinder(mean_eval.detach().cpu().numpy(), eval_dset.num_views, top_view=True)
    if run is not None:
        run["plots/embeddings_cylinder_top"].upload(fig)
    # endregion

    # region PLOT SUBMISSION
    unique_images = get_data_from_dataloader(dset_unique_loader, 0)
    x_rec = get_reconstructions(unique_mean, unique_extra, decoder, args.extra_dim, args.autoencoder)
    x_rec = sigmoid(x_rec.permute((0, 2, 3, 1)).detach().cpu().numpy())
    figures = plot_image_mixture_rec_all(unique_images, unique_mean, x_rec, num_objects_per_row=20, num_rows=7)
    for num_fig, fig in enumerate(figures):
        fig.savefig(os.path.join(save_folder, f"{num_fig}_img_mix_rec.png"), bbox_inches='tight')
        plt.close(fig)
        if run is not None:
            run[f"/plots/img_mix_rec{num_fig}"].upload(fig)
    # endregion

    # region LATENT TRAVERSAL
    num_unique_examples = 1
    # Select based on type of object
    dset_unique = ModelNetUniqueDataset(**train_data_parameters, index_unique_factors=-1)
    dset_unique_loader = torch.utils.data.DataLoader(dset_unique, batch_size=10, shuffle=False, num_workers=4,
                                                     pin_memory=True)

    unique_mean, unique_logvar, unique_std, unique_extra = get_embeddings(dset_unique_loader, model, args.variablescale, device)
    num_unique_orbits = len(unique_mean)

    if args.latent_dim != 4:
        num_points_traversal = 15
        num_objects = unique_mean.shape[0]
        if args.autoencoder != "None":
            angles_traversal = np.linspace(0, 1, num_points_traversal, endpoint=False) * 2 * np.pi
            angles_traversal = torch.tensor(angles_traversal).float()

            if args.latent_dim == 2:
                # Rotate the embeddings
                rotated_embeddings = []
                print("Angles traversal shape", angles_traversal.shape)
                rot = make_rotation_matrix(angles_traversal).unsqueeze(0)
                print("Rotation matrices shape", rot.shape, "Unique mean shape", unique_mean.shape)
                rotated_embeddings = (rot @ unique_mean.unsqueeze(-1).unsqueeze(1).float()).squeeze(-1)
                print("Rotated embeddings shape", rotated_embeddings.shape)
                # Rotated embeddings shape (num_points_traversal, num_unique, N, latent_dim)

            # Unique extra shape (num_unique, extra_dim)
            if args.extra_dim > 0:
                if args.autoencoder.startswith("vae"):
                    extra_repeated = unique_extra[:, -2 * args.extra_dim: args.extra_dim].unsqueeze(0)
                else:
                    # extra_repeated = unique_extra.unsqueeze(0)
                    extra_repeated = unique_extra
                print("Unique extra repeated shape", extra_repeated.shape)
                if num_objects > 1:
                    extra_repeated = extra_repeated.repeat((1, num_points_traversal, 1),
                                                           0)
                else:
                    extra_repeated = extra_repeated.repeat((num_points_traversal, 1), 0)
                extra_repeated = extra_repeated.view(-1, extra_repeated.shape[-1])
                print("Unique extra repeated shape", extra_repeated.shape)
                # mean_extra = mean_extra.repeat(num_points_traversal, 0)
                if args.autoencoder == "ae_single" or args.autoencoder == "vae":
                    rotated_embeddings = rotated_embeddings.permute((1, 0, 2, 3))
                    # rotated_embeddings = rotated_embeddings.view(num_points_traversal, num_objects, args.num,
                    #                                              args.latent_dim)
                    rotated_embeddings = rotated_embeddings[:, :, 0, :]
                    print("Rotated embeddings shape", rotated_embeddings.shape)
                    extra_repeated = extra_repeated.view(num_points_traversal, num_objects, extra_repeated.shape[-1])
                    print("Extra repeated shape", extra_repeated.shape)
                    z = torch.cat([rotated_embeddings, extra_repeated], dim=-1).float()
                    print("Z shape", z.shape)
                    z = z.view((-1, z.shape[-1]))
                else:
                    rotated_embeddings = rotated_embeddings.view(-1, rotated_embeddings.shape[-1] *
                                                                 rotated_embeddings.shape[-2])
                    z = torch.cat([rotated_embeddings, extra_repeated], dim=-1).float()

            else:
                if args.autoencoder == "ae_single":
                    z = rotated_embeddings.reshape(num_points_traversal * num_objects, args.num, args.latent_dim)[:, 0,
                        :]
                else:
                    z = rotated_embeddings

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
                run["plots/traversal"].upload(plt.gcf())
    # endregion

    # region ESTIMATE HIT-RATE

    print(f"Loading dataset {args.dataset} with dataset name {args.dataset_name}")

    # Note that the batch size is fixed to 20
    eval_batch_size = 20
    if args.dataset.startswith("modelnet"):
        eval_dset = ModelNetDataset("/data/volume_2/data/active_views",
                                    split="train",
                                    object_type_list=args.dataset_name,
                                    examples_per_object=12,
                                    use_random_initial=True,
                                    total_views=360,
                                    fixed_number_views=12,
                                    shuffle_available_views=True,
                                    use_random_choice=False,
                                    seed=70)
    elif args.dataset.startswith("shrec21shape"):
        eval_dset = ModelNetDataset("./data/shrec21shape",
                                    split="train",
                                    object_type_list=args.dataset_name,
                                    examples_per_object=12,
                                    use_random_initial=True,
                                    total_views=12,
                                    fixed_number_views=12,
                                    shuffle_available_views=True,
                                    use_random_choice=False,
                                    seed=70)


    val_loader = torch.utils.data.DataLoader(eval_dset, batch_size=eval_batch_size, shuffle=True, num_workers=4,
                                             pin_memory=True)
    print("# test set:", len(eval_dset))
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
            print("Action, mean shape", action.shape, z_mean.shape)
            z_mean_pred = action @ z_mean
        elif args.latent_dim > 3 and args.latent_dim % 2 == 0:

            action = action.squeeze(1)
            z_mean_pred = so2_rotate_subspaces(z_mean, action, detach=False)
        else:
            raise ValueError(f"Rotation not defined for latent dimension {args.latent_dim} ")

        chamfer_matrix = \
            ((z_mean_pred.unsqueeze(1).unsqueeze(1) - z_mean_next.unsqueeze(2).unsqueeze(0)) ** 2).sum(-1).sum(
                -1).min(
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
    num_unique_examples = 1
    if n_data_elements == 5:
        identifiers = dset.factors[0]
    else:
        identifiers = dset.stabs
    num_unique_orbits = len(np.unique(identifiers))
    for unique in np.unique(dset.stabs):
        unique_images.append(dset.data[identifiers == unique][:num_unique_examples, 0])

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
                if num_unique_examples == 1:
                    add_image_to_ax(1 / (1 + np.exp(-rec)), axes)
                    add_image_to_ax(unique_images[j, k].permute((1, 2, 0)).detach().cpu().numpy(), axes2)
                else:
                    add_image_to_ax(1 / (1 + np.exp(-rec)), axes[k])
                    add_image_to_ax(unique_images[j, k].permute((1, 2, 0)).detach().cpu().numpy(), axes2[k])
            fig.savefig(os.path.join(save_folder, f"reconstruction_{j}.png"), bbox_inches='tight')
            if run is not None:
                run["plots/reconstruction" + str(j)].upload(fig)
            if run is not None:
                run["plots/input_image" + str(j)].upload(fig2)
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
            run["plots/probabilities_" + str(j)].upload(plt.gcf())
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
            run["plots/rotvec" + str(j)].upload(plt.gcf())
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
            run["plots/rotvec_alone" + str(j)].upload(plt.gcf())
        fig.savefig(os.path.join(save_folder, f"rotvec_alone_{j}.png"), bbox_inches='tight')

    print(f"Loading dataset {args.dataset} with dataset name {args.dataset_name}")
    dset_val = EquivDataset(f'{args.data_dir}/{args.dataset}/',
                            list_dataset_names=[dataset_name + "_val" for dataset_name in args.dataset_name],
                            max_data_per_dataset=-1)

    # Note that the batch size is fixed to 20
    eval_batch_size = 20
    val_loader = torch.utils.data.DataLoader(dset_val, batch_size=eval_batch_size, shuffle=True, num_workers=4,
                                             pin_memory=True)
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

        print("Z mean pred shape", z_mean_pred.shape)
        chamfer_matrix = \
            ((z_mean_pred.unsqueeze(1).unsqueeze(1) - z_mean_next.unsqueeze(2).unsqueeze(0)) ** 2).sum(-1).sum(
                -1).min(
                dim=-1)[
                0].mean(dim=-1)

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
if run is not None:
    run["sys/failed"] = False
    run.stop()
