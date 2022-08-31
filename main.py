from utils.parse_args import get_args

parser = get_args()
args = parser.parse_args()

if args.use_comet:
    import utils.comet_config
    from comet_ml import Experiment

    experiment = Experiment(**utils.comet_config.experiment_params)
    hyperparameters = vars(args)
    experiment.log_parameters(hyperparameters)

import pickle
from models.resnet import *

# Import datasets
from datasets.equiv_dset import *

from models.models_nn import *
from utils.nn_utils import *
import torch.utils.data
from torch import save
from utils.plotting_utils import plot_embeddings_eval
import matplotlib.pyplot as plt

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
if args.optimizer == "adam":
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
elif args.optimizer == "adamw":
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
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

    for batch_idx, (img, img_next, action) in enumerate(data_loader):
        batch_size = img.shape[0]
        global_step += batch_size

        if mode == 'train':
            optimizer.zero_grad()
            model.train()

        elif mode == 'val':
            model.eval()

        img = img.to(device)
        img_next = img_next.to(device)
        action = action.to(device).squeeze(1)
        z_mean, z_logvar, extra = model(img)
        z_mean_next, z_logvar_next, extra_next = model(img_next)
        z_logvar = -4.6 * torch.ones(z_logvar.shape).to(z_logvar.device)
        z_logvar_next = -4.6 * torch.ones(z_logvar.shape).to(z_logvar.device)

        rot = make_rotation_matrix(action)

        # The predicted z_mean after applying the rotation corresponding to the action
        # z_mean_pred = (rot @ z_mean.unsqueeze(-1)).squeeze(-1)  # Beware the detach!!!
        z_mean_pred = (rot @ z_mean.unsqueeze(-1).detach()).squeeze(-1)  # Beware the detach!!!

        # Probabilistic losses (cross-entropy, Chamfer etc)
        # loss_equiv = prob_loss(z_mean_next, z_logvar_next, z_mean_pred, z_logvar, N)
        # loss_equiv = prob_loss(z_mean_pred, z_logvar, z_mean_next, z_logvar_next, N)
        loss_equiv = ((z_mean_pred.unsqueeze(1) - z_mean_next.unsqueeze(2)) ** 2).sum(-1).min(dim=-1)[0].sum(
            dim=-1).mean()  # Chamfer/Hausdorff loss

        # loss_equiv = ((z_mean_pred - z_mean_next)**2).sum(-1).mean()

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

            loss = loss_equiv + loss_contra

            if args.use_comet:
                experiment.log_metric("batch_loss_equiv", loss_equiv.cpu().numpy())
                experiment.log_metric("batch_loss_contra", loss_contra.cpu().numpy())

        else:
            loss = loss_equiv
            if args.use_comet:
                experiment.log_metric("batch_loss_equiv", loss_equiv.cpu().numpy())

        print(
            f"{mode.upper()} Epoch: {epoch}, Batch: {batch_idx} of {len(data_loader)} Loss: {loss:.3} Loss equiv:"
            f" {loss_equiv:.3}")

        mu_loss += loss.item()

        if mode == 'train':
            loss.backward()
            optimizer.step()

    mu_loss /= len(data_loader)
    if args.use_comet:
        experiment.log_metric("loss", mu_loss)

    if mode == 'val':
        errors.append(mu_loss)
        np.save(f'{MODEL_PATH}/errors_val.npy', errors)

        if (epoch % args.save_interval) == 0:
            save(model, model_file)
        if (args.plot > 0) & ((epoch % (args.plot+1)) == 0):
            mean_eval, logvar_eval, extra_eval = model(eval_images.to(device))
            logvar_eval = -4.6 * torch.ones(logvar_eval.shape).to(logvar_eval.device)
            std_eval = np.exp(logvar_eval.detach().cpu().numpy() / 2.) / 10
            for num_unique, unique in enumerate(np.unique(stabilizers)):
                boolean_selection = (stabilizers == unique)
                if args.dataset_name[0].endswith("m"):
                    print("Plotting stabilizers equal to 1")
                    fig, axes = plot_embeddings_eval(mean_eval[boolean_selection], std_eval[boolean_selection], N,
                                                     np.ones_like(stabilizers[boolean_selection]))
                    axes.set_xlim([-1.2, 1.2])
                    axes.set_ylim([-1.2, 1.2])
                else:
                    fig, axes = plot_embeddings_eval(mean_eval[boolean_selection], std_eval[boolean_selection], N,
                                                     stabilizers[boolean_selection])
                    axes.set_xlim([-1.2, 1.2])
                    axes.set_ylim([-1.2, 1.2])
                axes.set_title(f"Target stabilizers = {unique}")
                fig.savefig(os.path.join(os.path.join(MODEL_PATH, "figures"), f"{unique}_{epoch}.png"), bbox_inches='tight')
                plt.close(fig)


if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch, train_loader, 'train')
        with torch.no_grad():
            train(epoch, val_loader, 'val')
