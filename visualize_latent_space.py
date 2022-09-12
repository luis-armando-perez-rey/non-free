import argparse
import pickle
from scipy.spatial.transform import Rotation as R

import matplotlib.pyplot as plt
import torch.utils.data
from torch import load

# Import datasets
from datasets.equiv_dset import *
from utils.nn_utils import *

# Import plotting utils
from utils.plotting_utils import plot_extra_dims, plot_images_distributions, plot_embeddings_eval, \
    save_embeddings_on_circle, plot_images_reconstructions, plot_images_multi_reconstructions, load_plot_val_errors, \
    plot_embeddings_eval_torus, yiq_embedding

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
    stabilizers = None
    eval_images = None
elif args.dataset == 'platonics':
    dset = PlatonicMerged(N=30000, data_dir=args.data_dir)
    stabilizers = None
elif args.dataset == "arrows" or args.dataset == "sinusoidal":
    dset = EquivDatasetStabs(f'{args.data_dir}/{args.dataset}/', list_dataset_names=args.dataset_name)
    dset_eval = EvalDataset(f'{args.data_dir}/{args.dataset}/', list_dataset_names=args.dataset_name)
    eval_images = torch.FloatTensor(dset_eval.data.reshape(-1, *dset_eval.data.shape[2:]))
    stabilizers = dset_eval.stabs.reshape((-1))

elif args.dataset.endswith("translation"):
    dset = EquivDatasetStabs(f'{args.data_dir}/{args.dataset}/', list_dataset_names=args.dataset_name)
    dset_eval = EvalDataset(f'{args.data_dir}/{args.dataset}/', list_dataset_names=args.dataset_name)
    eval_images = torch.FloatTensor(dset_eval.data.reshape(-1, *dset_eval.data.shape[2:]))
    stabilizers = dset_eval.stabs.reshape((-1))
    eval_actions = dset_eval.lbls.reshape((-1, 2))
else:
    eval_images = None
    stabilizers = None
    raise ValueError(f'Dataset {args.dataset} not supported')

train_loader = torch.utils.data.DataLoader(dset,
                                           batch_size=20, shuffle=True)
device = 'cpu'
model = load(model_file).to(device)
if args.autoencoder != 'None':
    decoder = load(decoder_file).to(device)
model.eval()

if args.dataset == "arrows" or args.dataset == "sinusoidal" or args.dataset.endswith("translation"):
    img, img_next, action, n_stabilizers = next(iter(train_loader))
    print("EVAL IMAGES SHAPE", eval_images.shape)
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
fig.savefig(os.path.join(save_folder, 'invariant.png'))

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
extra = extra.detach().cpu().numpy()

if args.latent_dim == 2 or args.latent_dim == 4:
    if args.latent_dim == 2:
        rot = make_rotation_matrix(action)
        mean_rot = (rot @ mean.unsqueeze(-1)).squeeze(-1)
        mean_rot = mean_rot.detach().cpu().numpy()
    else:
        action = action.squeeze(1)
        mean_rot = so2_rotate_subspaces(mean, action, detach=True)

    print(extra.shape)

    action = action.detach().cpu().numpy()

    # region PLOT IDENTITY EMBEDDINGS
    fig, ax = plot_extra_dims(extra, color_labels=n_stabilizers)
    if fig:
        fig.savefig(os.path.join(save_folder, 'invariant.png'))
    # endregion

    # region PLOT ROTATED EMBEDDINGS
    for i in range(10):
        print(f"Plotting example {i}")
        fig, axes = plot_images_distributions(mean=mean_numpy[i], std=std[i], mean_next=mean_next[i],
                                              std_next=std_next[i],
                                              image=npimages[i], image_next=npimages_next[i],
                                              expected_mean=mean_rot[i], n=N)
        fig.savefig(os.path.join(save_folder, f"test_{i}.png"), bbox_inches='tight')
        plt.close("all")
    # endregion
    # Save the plots of the embeddings on the circle
    # save_embeddings_on_circle(mean_eval, std_eval, stabilizers, save_folder, args.dataset_name[0])

    # # region PLOT EVALUATION DATASET EMBEDDINGS
    # for num_unique, unique in enumerate(np.unique(stabilizers)):
    #     boolean_selection = (stabilizers == unique)
    #     if args.dataset_name[0].endswith("m"):
    #         print("Plotting stabilizers equal to 1")
    #         plot_stabilizers = np.ones_like(stabilizers[boolean_selection])
    #     else:
    #         plot_stabilizers = stabilizers[boolean_selection]
    #
    #     fig, axes = plot_embeddings_eval(mean_eval[boolean_selection], std_eval[boolean_selection], N,
    #                                      plot_stabilizers, increasing_radius=True)
    #     axes.set_title(f"Target stabilizers = {unique}")
    #     fig.savefig(os.path.join(save_folder, f"{unique}_eval_radius.png"), bbox_inches='tight')
    # # endregion

    # region PLOT ON THE TORUS
    if args.latent_dim == 4:
        print("EVAL ACTIONS SHAPE", eval_actions.shape)
        colors = yiq_embedding(eval_actions[:, 0], eval_actions[:, 1])
        print("COLORS SHAPE", colors.shape)
        fig, ax = plot_embeddings_eval_torus(mean_eval.detach().numpy(), colors)
        fig.savefig(os.path.join(save_folder, f"torus_eval_embedings.png"), bbox_inches='tight')
        plt.close("all")
    # endregion

    # region PLOT RECONSTRUCTIONS
    if args.autoencoder != "None":
        reconstructions = []

        for n in range(N):
            if args.extra_dim > 0:
                x_rec = decoder(torch.concat([mean_eval[:, n], extra_eval], dim=-1))
            else:
                x_rec = decoder(mean_eval[:, n])
            x_rec = torch.permute(x_rec, (0, 2, 3, 1))
            reconstructions.append(x_rec)
        reconstructions = torch.stack(reconstructions, dim=1)
        reconstructions_np = reconstructions.detach().cpu().numpy()
        for num_unique, unique in enumerate(np.unique(stabilizers)):
            boolean_selection = (stabilizers == unique)
            fig, _ = plot_images_multi_reconstructions(npimages_eval[boolean_selection][:5],
                                                       reconstructions_np[boolean_selection][:5])
            fig.savefig(os.path.join(save_folder, f"{unique}_reconstructions.png"), bbox_inches='tight')
    # endregion

elif args.latent_dim == 3:
    for i in range(10):
        plt.figure()
        plt.imshow(npimages_eval[i])
        plt.figure()
        ax = plt.axes(projection='3d')
        rots = R.from_dcm(mean_numpy[i]).as_rotvec()  # .as_euler('zxy', degrees=False)
        ax.scatter3D(rots[:, 0], rots[:, 1], rots[:, 2], s=[30] * len(rots))
        plt.show()
