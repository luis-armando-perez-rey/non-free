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
if args.autoencoder != "None":
    dec = Decoder(nc=img_shape[0], latent_dim=2, extra_dim=extra_dim, model=args.autoencoder).to(device)
    parameters += list(dec.parameters())
else:
    dec = None

if args.optimizer == "adam":
    optimizer = optim.Adam(parameters, lr=args.lr)
elif args.optimizer == "adamw":
    optimizer = optim.AdamW(parameters, lr=args.lr)
elif args.optimizer == "sgd":
    optimizer = optim.SGD(parameters, lr=args.lr, momentum=0.9, nesterov=True)
else:
    ValueError(f"Optimizer {args.optimizer} not defined")

errors = []


def make_rotation_matrix(action):
    s = torch.sin(action)
    c = torch.cos(action)
    rot = torch.stack([torch.stack([c, -s]), torch.stack([s, c])])
    rot = rot.permute((2, 0, 1)).unsqueeze(1)
    return rot


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
        z_logvar = -4.6 * torch.ones(z_logvar.shape).to(z_logvar.device)
        z_logvar_next = -4.6 * torch.ones(z_logvar.shape).to(z_logvar.device)

        rot = make_rotation_matrix(action)

        # The predicted z_mean after applying the rotation corresponding to the action
        # z_mean_pred = (rot @ z_mean.unsqueeze(-1)).squeeze(-1)  # Beware the detach!!!
        z_mean_pred = (rot @ z_mean.unsqueeze(-1).detach()).squeeze(-1)  # Beware the detach!!!


        # Chamfer distance is used for validation
        if mode == "train":
            equiv_loss_type = args.equiv_loss
        elif mode == "val":
            equiv_loss_type = "chamfer"

        # Probabilistic losses (cross-entropy, Chamfer etc)
        if equiv_loss_type == "cross-entropy":
            loss_equiv = prob_loss(z_mean_pred, z_logvar, z_mean_next, z_logvar_next, N)
        elif equiv_loss_type == "vm-cross-entropy":
            loss_equiv = prob_loss_vm(z_mean_pred, z_logvar, z_mean_next, z_logvar_next, N)
        elif equiv_loss_type == "chamfer":
            loss_equiv = ((z_mean_pred.unsqueeze(1) - z_mean_next.unsqueeze(2)) ** 2).sum(-1).min(dim=-1)[0].sum(
                dim=-1).mean()  # Chamfer/Hausdorff loss
        elif equiv_loss_type == "euclidean":
            loss_equiv = ((z_mean_pred - z_mean_next) ** 2).sum(-1).mean()
        else:
            loss_equiv = 0
            ValueError(f"{args.equiv_loss} not available")

        # loss = loss_equiv
        losses = [loss_equiv]
        if extra_dim > 0:
            if args.identity_loss == "infonce":
                # Contrastive Loss: infoNCE w/ cosine similariy
                distance_matrix = (extra.unsqueeze(1) * extra_next.unsqueeze(0)).sum(-1) / args.tau
                # print("Extra shape", extra.shape)
                loss_contra = -torch.mean(
                    (extra * extra_next).sum(-1) / args.tau - torch.logsumexp(distance_matrix, dim=-1))
            elif args.identity_loss == "euclidean":
                loss_contra = 1000.0 * torch.mean(torch.sum((extra - extra_next) ** 2, dim=-1))
            else:
                loss_contra = 0.0
            losses.append(loss_contra)

        # Reconstruction
        reconstruction_loss = 0
        if args.autoencoder != "None":
            for n in range(N):
                x_rec = dec(z_mean[:, n])
                x_next_rec = dec(z_mean_next[:, n])
                reconstruction_loss += torch.square(image - x_rec).sum(-1).mean()
                reconstruction_loss += torch.square(img_next - x_next_rec).sum(-1).mean()
            losses.append(reconstruction_loss)

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
