from utils.parse_args import get_args
import pickle
import torch.utils.data
from torch import save
import sys
# Import datasets
from datasets.equiv_dset import *
from models.models_nn import *
from utils.nn_utils import *
from utils.plotting_utils import save_embeddings_on_circle
from models.losses import EquivarianceLoss, ReconstructionLoss, IdentityLoss, estimate_entropy
from models.distributions import MixtureDistribution, get_prior, get_z_values

# region PARSE ARGUMENTS
parser = get_args()
args = parser.parse_args()
print(args)
# endregion

# region TORCH SETUP
# Print set up torch device, empty cache and set random seed
torch.cuda.empty_cache()
torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device: ', device)
# endregion

# region PATHS
model_path = os.path.join(args.checkpoints_dir, args.model_name)
figures_dir = os.path.join(model_path, 'figures')
model_file = os.path.join(model_path, 'model.pt')
decoder_file = os.path.join(model_path, "decoder.pt")
meta_file = os.path.join(model_path, 'metadata.pkl')
log_file = os.path.join(model_path, 'log.txt')
make_dir(model_path)
make_dir(figures_dir)
# endregion

# Save the arguments
pickle.dump({'args': args}, open(meta_file, 'wb'))

# region SET DATASET
if args.dataset == 'platonics':
    dset = PlatonicMerged(N=30000, data_dir=args.data_dir)
else:
    print(f"Loading dataset {args.dataset} with dataset name {args.dataset_name}")
    dset = EquivDataset(f'{args.data_dir}/{args.dataset}/', list_dataset_names=args.dataset_name)
    # NOTICE THAT I REMOVED LOADING OF LABELS FOR EVAL
    dset_eval = EvalDataset(f'{args.data_dir}/{args.dataset}/', list_dataset_names=args.dataset_name)
    eval_images = torch.FloatTensor(dset_eval.data.reshape(-1, dset_eval.data.shape[-1]))
    stabilizers = dset_eval.stabs.reshape((-1))

# Setup torch dataset
dset, dset_test = torch.utils.data.random_split(dset, [len(dset) - int(len(dset) / 10), int(len(dset) / 10)])
train_loader = torch.utils.data.DataLoader(dset, batch_size=args.batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(dset_test, batch_size=args.batch_size, shuffle=True)
print("# train set:", len(dset))
print("# test set:", len(dset_test))

# Sample data
img, _, acn = next(iter(train_loader))
img_shape = img.shape[1:]
# endregion

# region SET MODEL
N = args.num  # number of Gaussians per group latent space
extra_dim = args.extra_dim  # the invariant component

print("Using model", args.model)
if args.use_simplified:
    model = MDNSimplified(img_shape[0], args.latent_dim, N, extra_dim, model=args.model, normalize_extra=True).to(device)
else:
    model = MDN(img_shape[0], args.latent_dim, N, extra_dim, model=args.model, normalize_extra=True).to(device)
parameters = list(model.parameters())
if (args.autoencoder != "None") & (args.decoder != "None"):
    dec = Decoder(nc=img_shape[0], latent_dim=args.latent_dim, extra_dim=extra_dim,
                  model=args.decoder).to(
        device)
    parameters += list(dec.parameters())
    rec_loss_function = ReconstructionLoss(args.reconstruction_loss)
else:
    dec = None
# endregion

optimizer = get_optimizer(args.optimizer, args.lr, parameters)
errors = []
entropy = []

# region LOSS FUNCTIONS
identity_loss_function = IdentityLoss(args.identity_loss, temperature=args.tau)
equiv_loss_train_function = EquivarianceLoss(args.equiv_loss)
equiv_loss_val_function = EquivarianceLoss("chamfer_val")


# endregion

def matrix_dist(z_mean_next, z_mean_pred, latent_dim):
    if latent_dim == 2:
        return ((z_mean_next.unsqueeze(2) - z_mean_pred.unsqueeze(1)) ** 2).sum(-1)
    elif latent_dim == 3:
        return ((z_mean_next.unsqueeze(2) - z_mean_pred.unsqueeze(1)) ** 2).sum(-1).sum(-1)


def train(epoch, data_loader, mode='train'):
    mu_loss = 0
    global_step = len(data_loader) * epoch

    # Chamfer distance is used for validation
    if mode == "train":
        equiv_loss_function = equiv_loss_train_function
    elif mode == "val":
        equiv_loss_function = equiv_loss_val_function
    else:
        raise ValueError(f"Mode {mode} not defined")

    for batch_idx, (image, img_next, action) in enumerate(data_loader):
        batch_size = image.shape[0]
        global_step += batch_size

        if mode == 'train':
            optimizer.zero_grad()
            model.train()
            if args.autoencoder != "None":
                dec.train()

        elif mode == 'val':
            model.eval()

        image = image.to(device)
        img_next = img_next.to(device)
        action = action.to(device)

        # z_mean and z_mean_next have shape (batch_size, n, latent_dim)
        z_mean, z_logvar, extra = model(image)
        z_mean_next, z_logvar_next, extra_next = model(img_next)
        # TODO: Allow for different scale values
        # WARNING!! Notice that the scale parameter is being fixed!!!
        z_logvar = -4.6 * torch.ones(z_logvar.shape).to(z_logvar.device)
        z_logvar_next = -4.6 * torch.ones(z_logvar.shape).to(z_logvar.device)

        # Rotate the embeddings in Z_G of the first image by the action
        if args.latent_dim == 2:
            action = action.squeeze(1)
            rot = make_rotation_matrix(action)
            # The predicted z_mean after applying the rotation corresponding to the action
            z_mean_pred = (rot @ z_mean.unsqueeze(-1).detach()).squeeze(-1)  # Beware the detach!!!
        elif args.latent_dim == 3:
            z_mean_pred = action.unsqueeze(1) @ z_mean.detach()
        elif args.latent_dim >= 3 and args.latent_dim % 2 == 0:
            action = action.squeeze(1)
            z_mean_pred = so2_rotate_subspaces(z_mean, action, detach=True)
        else:
            raise ValueError(f"Rotation not defined for latent dimension {args.latent_dim} ")

        # Calculate equivariance loss
        p = MixtureDistribution(z_mean, z_logvar, args.enc_dist)
        p_next = MixtureDistribution(z_mean_next, z_logvar_next, args.enc_dist)
        p_pred = MixtureDistribution(z_mean_pred, z_logvar, args.enc_dist)
        loss_equiv = equiv_loss_function(p_pred, p_next)
        losses = [loss_equiv]

        # region CALCULATE IDENTITY LOSS
        if extra_dim > 0:
            loss_identity = identity_loss_function(extra, extra_next)
            losses.append(loss_identity)
        # endregion

        # region CALCULATE RECONSTRUCTION LOSS
        reconstruction_loss = 0
        if args.autoencoder != "None":
            # Get latent variables
            z = get_z_values(p, extra, args.num, args.autoencoder)
            z_next = get_z_values(p_next, extra_next, args.num, args.autoencoder)
            # Calculate reconstruction loss for each of the latent representations
            for n in range(args.num):
                x_rec = dec(z[:, n])
                x_next_rec = dec(z_next[:, n])
                reconstruction_loss += rec_loss_function(x_rec, image).mean()
                reconstruction_loss += rec_loss_function(x_next_rec, img_next).mean()
            losses.append(reconstruction_loss)
        # endregion

        # region CALCULATE KL LOSS
        if (args.autoencoder == "vae") and (
                (args.enc_dist == "von-mises-mixture") or (args.enc_dist == "gaussian-mixture")):
            # Get prior for VAE model
            prior = get_prior(z_mean.shape[0], args.num, 2, args.prior_dist, device)
            kl_loss = p.approximate_kl(prior) + p_next.approximate_kl(prior)
            losses.append(kl_loss)
        # endregion

        # Sum all losses
        if mode == "train":
            loss = sum(losses)
        elif mode == "val":
            loss = loss_equiv

        # region LOGGING LOSS
        # Print loss progress
        if args.autoencoder == "None":
            print(
                f"{mode.upper()} Epoch: {epoch}, Batch: {batch_idx} of {len(data_loader)} Loss: {loss:.3f} Loss equiv:"
                f" {loss_equiv:.3}")
        elif args.autoencoder == "ae":
            print(
                f"{mode.upper()} Epoch: {epoch}, Batch: {batch_idx} of {len(data_loader)} Loss: {loss:.3f} Loss equiv:"
                f" {loss_equiv:.3} Loss reconstruction: {reconstruction_loss:.3}")
        elif args.autoencoder == "vae":
            print(
                f"{mode.upper()} Epoch: {epoch}, Batch: {batch_idx} of {len(data_loader)} Loss: {loss:.3f} Loss equiv:"
                f" {loss_equiv:.3} Loss reconstruction: {reconstruction_loss:.3} Loss KL: {kl_loss:.3}")
        # endregion

        mu_loss += loss.item()

        if mode == 'train':
            loss.backward()
            optimizer.step()

    mu_loss /= len(data_loader)

    if mode == 'val':
        errors.append(mu_loss)
        entropy.append(estimate_entropy(p, 1000).item())
        np.save(f'{model_path}/errors_val.npy', errors)
        np.save(f'{model_path}/entropy_val.npy', entropy)

        if (epoch % args.save_interval) == 0:
            save(model, model_file)
            if args.autoencoder != "None":
                save(dec, decoder_file)
        if (args.plot > 0) & ((epoch % (args.plot + 1)) == 0):
            mean_eval, logvar_eval, extra_eval = model(eval_images.to(device))
            logvar_eval = -4.6 * torch.ones(logvar_eval.shape).to(logvar_eval.device)
            std_eval = np.exp(logvar_eval.detach().cpu().numpy() / 2.) / 10
            plot_save_folder = os.path.join(os.path.join(model_path, "figures"))
            save_embeddings_on_circle(mean_eval, std_eval, stabilizers, plot_save_folder)

    # Write to standard output. Allows printing to file progress when using HPC
    sys.stdout.flush()


if __name__ == "__main__":
    for epoch_ in range(1, args.epochs + 1):
        train(epoch_, train_loader, 'train')
        with torch.no_grad():
            train(epoch_, val_loader, 'val')
