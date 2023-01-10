from utils.parse_args import get_args
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
if args.neptune_user != "":
    from utils.neptune_utils import initialize_neptune_run, save_sys_id

    run = initialize_neptune_run(args.neptune_user, "non-free")
    run["parameters"] = vars(args)
else:
    run = None
print(args)

# endregion

# region TORCH SETUP
# Print set up torch device, empty cache and set random seed
torch.cuda.empty_cache()
# torch.manual_seed(args.seed)
device = torch.device("cuda:" + args.gpu if torch.cuda.is_available() else "cpu")
print('Using device: ', device)
# endregion

# region PATHS
# Paths were the models will be saved
model_path = os.path.join(args.checkpoints_dir, args.model_name)
make_dir(model_path)
decoder_file = os.path.join(model_path, "decoder.pt")
# Paths for saving the images
figures_dir = os.path.join(model_path, 'figures')
make_dir(figures_dir)
model_file = os.path.join(model_path, 'model.pt')
meta_file = os.path.join(model_path, 'metadata.pkl')
# Neptune txt file
if run is not None:
    neptune_id_path = os.path.join(model_path, 'neptune.txt')
    save_sys_id(run, neptune_id_path)

# endregion

# region SAVE METADATA
# Save the arguments
pickle.dump({'args': args}, open(meta_file, 'wb'))
# endregion

# region SET DATASET
if args.dataset == 'platonics':
    dset = PlatonicMerged(N=30000, data_dir=args.data_dir)
else:
    print(f"Loading dataset {args.dataset} with dataset name {args.dataset_name}")
    dset = EquivDataset(f'{args.data_dir}/{args.dataset}/', list_dataset_names=args.dataset_name,
                        max_data_per_dataset=args.ndatapairs)

    dset_val = EquivDataset(f'{args.data_dir}/{args.dataset}/',
                            list_dataset_names=[dataset_name + "_val" for dataset_name in args.dataset_name],
                            max_data_per_dataset=-1)
    if args.dataset != "symmetric_solids":
        dset_eval = EvalDataset(f'{args.data_dir}/{args.dataset}/', list_dataset_names=args.dataset_name)
        eval_images = torch.FloatTensor(dset_eval.data.reshape(-1, dset_eval.data.shape[-1]))
        stabilizers = dset_eval.stabs.reshape((-1))

# Setup torch dataset
# dset, dset_val = torch.utils.data.random_split(dset, [len(dset) - int(len(dset) / 10), int(len(dset) / 10)])
train_loader = torch.utils.data.DataLoader(dset, batch_size=args.batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(dset_val, batch_size=args.batch_size, shuffle=True)

print("# train set:", len(dset))
print("# test set:", len(dset_val))

# Sample data
img, _, acn = next(iter(train_loader))
img_shape = img.shape[1:]
# endregion

# region SET MODEL
N = args.num  # number of Gaussians per group latent space
extra_dim = args.extra_dim  # the invariant component

print("Using model", args.model)
if args.use_simplified:
    model = MDNSimplified(img_shape[0], args.latent_dim, N, extra_dim, model=args.model, normalize_extra=True).to(
        device)
else:
    model = MDN(img_shape[0], args.latent_dim, N, extra_dim, model=args.model, normalize_extra=True).to(device)
parameters = list(model.parameters())

if args.autoencoder == "ae_single":
    dec_dim = args.latent_dim
else:
    if args.latent_dim == 2:
        dec_dim = 2 * N
    elif args.latent_dim == 3:
        dec_dim = 9 * N
    elif args.latent_dim == 4:
        dec_dim = 4 * N
    else:
        dec_dim = 0

if (args.autoencoder != "None") & (args.decoder != "None"):
    dec = Decoder(nc=img_shape[0], latent_dim=dec_dim, extra_dim=extra_dim, model=args.decoder).to(device)

    parameters += list(dec.parameters())
    rec_loss_function = ReconstructionLoss(args.reconstruction_loss)

else:
    dec = None
# endregion

optimizer = get_optimizer(args.optimizer, args.lr, parameters)
errors = []
errors_rec = []
entropy = []
equiv_errors = []
reconstruction_errors = []
invariant_loss = []

# region LOSS FUNCTIONS
identity_loss_function = IdentityLoss(args.identity_loss, temperature=args.tau)
equiv_loss_train_function = EquivarianceLoss(args.equiv_loss)
equiv_loss_val_function = EquivarianceLoss("chamfer_val")


# endregion


def train(epoch, data_loader, mode='train'):
    mu_loss = 0
    mu_rec_loss = 0
    mu_equiv_loss = 0
    mu_id_loss = 0
    global_step = len(data_loader) * epoch

    # Chamfer distance is used for validation
    if mode == "train":
        equiv_loss_function = equiv_loss_train_function
    elif mode == "val":
        equiv_loss_function = equiv_loss_val_function
    else:
        raise ValueError(f"Mode {mode} not defined")

    # Initialize the losses that will be used for logging with Neptune
    total_batches = len(data_loader)

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
        if not(args.variablescale):
            z_logvar = -4.6 * torch.ones(z_logvar.shape).to(z_logvar.device)
            z_logvar_next = -4.6 * torch.ones(z_logvar.shape).to(z_logvar.device)

        # Rotate the embeddings in Z_G of the first image by the action
        if args.latent_dim == 2:
            action = action.squeeze(1)
            rot = make_rotation_matrix(action)
            # The predicted z_mean after applying the rotation corresponding to the action
            z_mean_pred = (rot @ z_mean.unsqueeze(-1).detach()).squeeze(-1)  # Beware the detach!!!
        elif args.latent_dim == 3:
            # print(z_mean.shape)
            # z_mean_pred = action @ z_mean.detach()
            # NOTICE!!!! Removed the detachment
            z_mean_pred = action @ z_mean
        elif args.latent_dim > 3 and args.latent_dim % 2 == 0:
            action = action.squeeze(1)
            z_mean_pred = so2_rotate_subspaces(z_mean, action, detach=True)
        else:
            raise ValueError(f"Rotation not defined for latent dimension {args.latent_dim} ")

        # Calculate equivariance loss
        p = MixtureDistribution(z_mean, z_logvar, args.enc_dist)
        p_next = MixtureDistribution(z_mean_next, z_logvar_next, args.enc_dist)
        p_pred = MixtureDistribution(z_mean_pred, z_logvar, args.enc_dist)
        loss_equiv = args.weightequivariance * equiv_loss_function(p_next, p_pred)
        losses = [loss_equiv]
        mu_equiv_loss += loss_equiv.item()
        if run is not None:
            run[mode + "/batch/loss_equiv"].log(loss_equiv)

        # region CALCULATE IDENTITY LOSS
        if extra_dim > 0:
            loss_identity = identity_loss_function(extra, extra_next)
            losses.append(loss_identity)
            mu_id_loss += loss_equiv.item()
            if run is not None:
                run[mode + "/batch/loss_identity"].log(loss_identity)
        else:
            loss_identity = 0
        # endregion

        # region CALCULATE RECONSTRUCTION LOSS
        reconstruction_loss = torch.tensor(0, dtype=image.dtype).to(device)
        if args.autoencoder != "None":
            if args.autoencoder == "ae_single":
                """
                Decode each of the group representations separately. 
                """
                # Get latent variables
                z = get_z_values(p, extra, args.num, args.autoencoder)
                z_next = get_z_values(p_next, extra_next, args.num, args.autoencoder)
                # Make the extra dimension invariant
                z_next[..., -extra_dim:] = z[..., -extra_dim:]
                # Calculate reconstruction loss for each of the latent representations
                for n in range(args.num):
                    x_rec = dec(z[:, n].detach())
                    x_next_rec = dec(z_next[:, n].detach())
                    reconstruction_loss += rec_loss_function(x_rec, image).mean()
                    reconstruction_loss += rec_loss_function(x_next_rec, img_next).mean()
            else:
                x_rec = dec(torch.cat([z_mean.view((z_mean.shape[0], -1)), extra], dim=-1).detach())
                reconstruction_loss += rec_loss_function(x_rec, image).mean()
            # TODO: Review if this helps
            # x_rec_next = dec(torch.cat([z_mean_next.view((z_mean.shape[0], -1)), extra], dim=-1).detach())
            # reconstruction_loss += rec_loss_function(x_rec_next, img_next).mean()

            losses.append(reconstruction_loss)
            if run is not None:
                run[mode + "/batch/reconstruction"].log(reconstruction_loss)
        # endregion

        # region CALCULATE KL LOSS
        if (args.autoencoder == "vae") and (
                (args.enc_dist == "von-mises-mixture") or (args.enc_dist == "gaussian-mixture")):
            # Get prior for VAE model
            prior = get_prior(z_mean.shape[0], args.num, 2, args.prior_dist, device)
            kl_loss = p.approximate_kl(prior) + p_next.approximate_kl(prior)
            losses.append(kl_loss)
        else:
            kl_loss = None
        # endregion

        # Sum all losses
        if mode == "train":
            loss = sum(losses)
        elif mode == "val":
            loss = loss_equiv / args.weightequivariance
        else:
            loss = None
        if run is not None:
            run[mode + "/batch/loss"].log(loss)

        # region LOGGING LOSS
        # Print loss progress
        if args.autoencoder == "None":
            if extra_dim > 0:
                print(f"Epoch: {epoch} , Batch: {batch_idx} of {len(data_loader)} "
                      f"Loss: {loss:.3f}"
                      f"Loss equiv: {loss_equiv:.3f}\t"
                      f"Identity: {loss_identity:.3f}\t")
            else:
                print(
                    f"{mode.upper()} Epoch: {epoch}, Batch: {batch_idx} of {len(data_loader)} Loss: {loss:.3f} Loss "
                    f"equiv: "
                    f" {loss_equiv:.3}")
        elif args.autoencoder.startswith("ae"):
            if extra_dim > 0:
                print(f"Epoch: {epoch} , Batch: {batch_idx} of {len(data_loader)} "
                      f"Loss: {loss:.3f}"
                      f"Loss equiv: {loss_equiv:.3f}\t"
                      f"Identity: {loss_identity:.3f}\t"
                      f"Reconstruction: {reconstruction_loss:.3f}\t")
            else:
                print(
                    f"Epoch: {epoch}, Batch: {batch_idx} of {len(data_loader)} Loss: {loss:.3f} Loss "
                    f"equiv: "
                    f" {loss_equiv:.3} Loss reconstruction: {reconstruction_loss:.3}")
        elif args.autoencoder == "vae":
            if extra_dim > 0:
                print(
                    f"Epoch: {epoch}, Batch: {batch_idx} of {len(data_loader)} Loss: {loss:.3f} Loss "
                    f"equiv: "
                    f" {loss_equiv:.3f} Loss reconstruction: {reconstruction_loss:.3f} Loss KL: {kl_loss:.3f} "
                    f"Loss identity: {loss_identity:.3f}")
            else:
                print(
                    f" Epoch: {epoch}, Batch: {batch_idx} of {len(data_loader)} Loss: {loss:.3f} Loss "
                    f"equiv: "
                    f" {loss_equiv:.3} Loss reconstruction: {reconstruction_loss:.3} Loss KL: {kl_loss:.3}")
        # endregion

        mu_loss += loss.item()
        mu_rec_loss += reconstruction_loss.item()

        if mode == 'train':
            loss.backward()
            optimizer.step()

    # Get each loss component to be logged by Neptune
    mu_loss /= total_batches
    mu_rec_loss /= total_batches
    mu_equiv_loss /= total_batches
    mu_id_loss /= total_batches

    if mode == 'val':
        errors.append(mu_loss)
        equiv_errors.append(mu_equiv_loss)
        errors_rec.append(mu_rec_loss)
        # If the encoding distribution is not None
        if p.components is not None:
            entropy.append(estimate_entropy(p, 1000).item())
        if extra_dim > 0:
            invariant_loss.append(loss_identity.item())
            np.save(f'{model_path}/invariant_val.npy', invariant_loss)
        np.save(f'{model_path}/errors_val.npy', errors)
        np.save(f'{model_path}/equiv_val.npy', equiv_errors)
        np.save(f'{model_path}/entropy_val.npy', entropy)
        np.save(f'{model_path}/errors_rec_val.npy', errors_rec)

        if args.autoencoder != "None":
            reconstruction_errors.append(reconstruction_loss.item())
            np.save(f'{model_path}/reconstruction_val.npy', reconstruction_errors)

        if (epoch % args.save_interval) == 0:
            save(model, model_file)
            if args.autoencoder != "None":
                save(dec, decoder_file)
        if (args.plot > 0) & ((epoch % (args.plot + 1)) == 0):
            mean_eval, logvar_eval, extra_eval = model(eval_images.to(device))
            if not(args.variablescale):
                logvar_eval = -4.6 * torch.ones(logvar_eval.shape).to(logvar_eval.device)
            std_eval = np.exp(logvar_eval.detach().cpu().numpy() / 2.) / 10
            plot_save_folder = os.path.join(os.path.join(model_path, "figures"))
            save_embeddings_on_circle(mean_eval, std_eval, stabilizers, plot_save_folder)

    # Write to standard output. Allows printing to file progress when using HPC
    sys.stdout.flush()



    if run is not None:
        run[mode + "/epoch/loss"].log(mu_loss)
        if args.autoencoder != "None":
            run[mode + "/epoch/reconstruction"].log(mu_rec_loss)
        run[mode + "/epoch/equivariance"].log(mu_equiv_loss)
        if extra_dim > 0:
            run[mode + "/epoch/identity"].log(mu_id_loss)


if __name__ == "__main__":
    for epoch_ in range(1, args.epochs + 1):
        train(epoch_, train_loader, 'train')
        with torch.no_grad():
            train(epoch_, val_loader, 'val')
    run.stop()
