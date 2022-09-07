from utils.parse_args import get_args
import pickle
import torch.utils.data
from torch import save
import sys
# Import datasets
from datasets.equiv_dset import *
from models.models_nn import *
from utils.nn_utils import *
from utils.plotting_utils import save_embeddings_on_circle, load_plot_val_errors
from models.losses import EquivarianceLoss, ReconstructionLoss, IdentityLoss
from models.distributions import MixtureDistribution

parser = get_args()
args = parser.parse_args()

torch.cuda.empty_cache()

torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device: ', device)

# Save paths
MODEL_PATH = os.path.join(args.checkpoints_dir, args.model_name)

figures_dir = os.path.join(MODEL_PATH, 'figures')
model_file = os.path.join(MODEL_PATH, 'model.pt')
decoder_file = os.path.join(MODEL_PATH, "decoder.pt")
meta_file = os.path.join(MODEL_PATH, 'metadata.pkl')
log_file = os.path.join(MODEL_PATH, 'log.txt')

make_dir(MODEL_PATH)
make_dir(figures_dir)

pickle.dump({'args': args}, open(meta_file, 'wb'))
print(args)
# Set dataset
print(f"Loading dataset {args.dataset} with dataset name {args.dataset_name}")
if args.dataset == 'rot-square':
    dset = EquivDataset(f'{args.data_dir}/square/', list_dataset_names=args.dataset_name)
elif args.dataset == 'rot-arrows':
    dset = EquivDataset(f'{args.data_dir}/arrows/', list_dataset_names=args.dataset_name)
elif args.dataset == 'sinusoidal':
    dset = EquivDataset(f'{args.data_dir}/sinusoidal/', list_dataset_names=args.dataset_name)
    dset_eval = EvalDataset(f'{args.data_dir}/sinusoidal/', list_dataset_names=args.dataset_name)
    eval_images = torch.FloatTensor(dset_eval.data.reshape(-1, dset_eval.data.shape[-1]))
    stabilizers = dset_eval.stabs.reshape((-1))
else:
    raise ValueError(f'Dataset {args.dataset} not supported')

dset, dset_test = torch.utils.data.random_split(dset, [len(dset) - int(len(dset) / 10), int(len(dset) / 10)])
train_loader = torch.utils.data.DataLoader(dset,
                                           batch_size=args.batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(dset_test,
                                         batch_size=args.batch_size, shuffle=True)

print("# train set:", len(dset))
print("# test set:", len(dset_test))

# Sample data
img, _, acn = next(iter(train_loader))
img_shape = img.shape[1:]
N = args.num  # number of Gaussians for the mixture model
extra_dim = args.extra_dim  # the invariant component

print("Using model", args.model)
model = MDN(img_shape[0], 2, N, extra_dim, model=args.model, normalize_extra=True).to(device)
parameters = list(model.parameters())
if (args.autoencoder != "None") & (args.decoder != "None"):
    dec = Decoder(nc=img_shape[0], latent_dim=2, extra_dim=extra_dim, model=args.decoder).to(device)
    parameters += list(dec.parameters())
    rec_loss_function = ReconstructionLoss(args.reconstruction_loss)
else:
    dec = None

if args.optimizer == "adam":
    optimizer = torch.optim.Adam(parameters, lr=args.lr)
elif args.optimizer == "adamw":
    optimizer = torch.optim.AdamW(parameters, lr=args.lr)
elif args.optimizer == "sgd":
    optimizer = torch.optim.SGD(parameters, lr=args.lr, momentum=0.9, nesterov=True)
else:
    ValueError(f"Optimizer {args.optimizer} not defined")

errors = []


def make_rotation_matrix(action):
    s = torch.sin(action)
    c = torch.cos(action)
    rot = torch.stack([torch.stack([c, -s]), torch.stack([s, c])])
    rot = rot.permute((2, 0, 1)).unsqueeze(1)
    return rot


# Define loss functions
encoding_dist_constructor = get_encoding_distribution_constructor(args.enc_dist)
identity_loss_function = IdentityLoss(args.identity_loss, temperature=args.tau)
equiv_loss_train_function = EquivarianceLoss(args.equiv_loss)
equiv_loss_val_function = EquivarianceLoss("chamfer")


def train(epoch, data_loader, mode='train'):
    mu_loss = 0
    global_step = len(data_loader) * epoch

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
        action = action.to(device).squeeze(1)
        # z_mean and z_mean_next have shape (batch_size, n, latent_dim)
        z_mean, z_logvar, extra = model(image)
        z_mean_next, z_logvar_next, extra_next = model(img_next)
        # TODO: Allow for different scale values
        # WARNING!! Notice that the scale parameter is being fixed!!!
        z_logvar = -4.6 * torch.ones(z_logvar.shape).to(z_logvar.device)
        z_logvar_next = -4.6 * torch.ones(z_logvar.shape).to(z_logvar.device)

        rot = make_rotation_matrix(action)

        # The predicted z_mean after applying the rotation corresponding to the action
        z_mean_pred = (rot @ z_mean.unsqueeze(-1).detach()).squeeze(-1)  # Beware the detach!!!

        # Chamfer distance is used for validation
        if mode == "train":
            equiv_loss_function = equiv_loss_train_function
        elif mode == "val":
            equiv_loss_function = equiv_loss_val_function
        else:
            raise ValueError(f"Mode {mode} not defined")

        # Calculate equivariance loss
        p = MixtureDistribution(z_mean, z_logvar, args.enc_dist)
        p_next = MixtureDistribution(z_mean_next, z_logvar_next, args.enc_dist)
        p_pred = MixtureDistribution(z_mean_pred, z_logvar, args.enc_dist)
        loss_equiv = equiv_loss_function(p_pred, p_next)
        losses = [loss_equiv]

        # Calculate loss identity
        if extra_dim > 0:
            loss_identity = identity_loss_function(extra, extra_next)
            losses.append(loss_identity)

        # Calculate reconstruction loss
        reconstruction_loss = 0
        if args.autoencoder != "None":
            z, z_next = get_z_values(p, p_next, extra, extra_next, n_samples=args.num,
                                     autoencoder_type=args.autoencoder)
            for n in range(args.num):
                x_rec = dec(z[:, n])
                x_next_rec = dec(z_next[:, n])
                reconstruction_loss += rec_loss_function(x_rec, image).mean()
                reconstruction_loss += rec_loss_function(x_next_rec, img_next).mean()
            losses.append(reconstruction_loss)

        # TODO: Verify that this works
        # Calculate KL loss
        if (args.autoencoder == "vae") and (args.enc_dist == "von-mises-mixture"):
            # Define the prior as N Von Mises distributions randomly placed with small concentration
            concentration = 0.01
            prior = torch.distributions.von_mises.VonMises(
                torch.tensor(2 * np.pi) * torch.rand((batch_size, N)).to(device),
                torch.tensor(concentration) * torch.ones((batch_size, N)).to(device))
            posterior_samples = p.sample((100,))
            kl = (p.component_distribution.log_prob(posterior_samples) - prior.log_prob(posterior_samples)).mean()
            posterior_samples = p_next.sample((100,))
            kl_next = (p_next.component_distribution.log_prob(posterior_samples) - prior.log_prob(
                posterior_samples)).mean()
            losses.append(kl + kl_next)

        # Sum all losses
        if mode == "train":
            loss = sum(losses)
        elif mode == "val":
            loss = loss_equiv

        if args.autoencoder == "None":
            print(
                f"{mode.upper()} Epoch: {epoch}, Batch: {batch_idx} of {len(data_loader)} Loss: {loss:.3f} Loss equiv:"
                f" {loss_equiv:.3}")
        else:
            print(
                f"{mode.upper()} Epoch: {epoch}, Batch: {batch_idx} of {len(data_loader)} Loss: {loss:.3f} Loss equiv:"
                f" {loss_equiv:.3} Loss reconstruction: {reconstruction_loss:.3}")

        mu_loss += loss.item()

        if mode == 'train':
            loss.backward()
            optimizer.step()

    mu_loss /= len(data_loader)

    if mode == 'val':
        errors.append(mu_loss)
        np.save(f'{MODEL_PATH}/errors_val.npy', errors)

        if (epoch % args.save_interval) == 0:
            save(model, model_file)
            if args.autoencoder != "None":
                save(dec, decoder_file)
        if (args.plot > 0) & ((epoch % (args.plot + 1)) == 0):
            mean_eval, logvar_eval, extra_eval = model(eval_images.to(device))
            logvar_eval = -4.6 * torch.ones(logvar_eval.shape).to(logvar_eval.device)
            std_eval = np.exp(logvar_eval.detach().cpu().numpy() / 2.) / 10
            plot_save_folder = os.path.join(os.path.join(MODEL_PATH, "figures"))
            save_embeddings_on_circle(mean_eval, std_eval, stabilizers, plot_save_folder)

    # Write to standard output. Allows printing to file progress when using HPC TUe
    sys.stdout.flush()


if __name__ == "__main__":
    for epoch_ in range(1, args.epochs + 1):
        train(epoch_, train_loader, 'train')
        with torch.no_grad():
            train(epoch_, val_loader, 'val')
    # Plot and save the validation errors
    fig, _ = load_plot_val_errors(f'{MODEL_PATH}/errors_val.npy')
    fig.savefig(f'{MODEL_PATH}/errors_val.png')
