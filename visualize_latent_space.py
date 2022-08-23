import argparse
import pickle

import matplotlib.pyplot as plt
import torch.utils.data
from matplotlib.patches import Ellipse
from torch import load

# Import datasets
from datasets.equiv_dset import *
from utils.nn_utils import *

# Import plotting utils
from utils.plotting_utils import plot_extra_dims, plot_images_distributions, plot_embeddings_eval

parser = argparse.ArgumentParser()
parser.add_argument('--save-folder', type=str, default='checkpoints', help='Path to saved model')
parser.add_argument('--dataset', type=str, default='dataset', help='Dataset')
parser.add_argument('--dataset_name', type=str, default='4', help='Dataset name')
args_eval = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.manual_seed(42)
torch.cuda.empty_cache()

model_file = os.path.join(args_eval.save_folder, 'model.pt')
meta_file = os.path.join(args_eval.save_folder, 'metadata.pkl')
args = pickle.load(open(meta_file, 'rb'))['args']

print(args)
if args.dataset == 'rot-square':
    dset = EquivDataset(f'{args.data_dir}/square/', dataset_name=args.dataset_name)
elif args.dataset == 'rot-arrows':
    print(args.dataset_name)
    dset = EquivDatasetStabs(f'{args.data_dir}/arrows/', dataset_name=args.dataset_name)
    dset_eval = EvalDataset(f'{args.data_dir}/arrows/', dataset_name=args.dataset_name)
    eval_images = torch.FloatTensor(dset_eval.data.reshape(-1, *dset_eval.data.shape[-3:]))
    stabilizers = dset_eval.stabs
else:
    raise ValueError(f'Dataset {args.dataset} not supported')

train_loader = torch.utils.data.DataLoader(dset,
                                           batch_size=100, shuffle=True)
device = 'cpu'
model = load(model_file).to(device)
model.eval()

if args.dataset == "rot-arrows":
    img, img_next, action, n_stabilizers = next(iter(train_loader))
    mean_eval, logvar_eval, extra_eval = model(eval_images.to(device))
    logvar_eval = -4.6 * torch.ones(logvar_eval.shape).to(logvar_eval.device)
    std_eval = np.exp(logvar_eval.detach().cpu().numpy() / 2.) / 10
else:
    img, img_next, action = next(iter(train_loader))
    n_stabilizers = None

img_shape = img.shape[1:]
npimages = np.transpose(img.detach().cpu().numpy(), axes=[0, 2, 3, 1])
npimages_next = np.transpose(img_next.detach().cpu().numpy(), axes=[0, 2, 3, 1])

mean, logvar, extra = model(img.to(device))
mean_next, logvar_next, extra_next = model(img_next.to(device))

logvar = -4.6 * torch.ones(logvar.shape).to(logvar.device)
logvar_next = -4.6 * torch.ones(logvar.shape).to(logvar.device)

s = torch.sin(action.squeeze(-1))
c = torch.cos(action.squeeze(-1))
rot = torch.stack([torch.stack([c, -s]), torch.stack([s, c])]).permute((2, 0, 1)).unsqueeze(1)
mean_rot = (rot @ mean.unsqueeze(-1)).squeeze(-1)

mean = mean.detach().cpu().numpy()
mean_next = mean_next.detach().cpu().numpy()
mean_rot = mean_rot.detach().cpu().numpy()
std = np.exp(logvar.detach().cpu().numpy() / 2.) / 10
std_next = np.exp(logvar_next.detach().cpu().numpy() / 2.) / 10
extra = extra.detach().cpu().numpy()

action = action.detach().cpu().numpy()

N = args.num
extra_dim = args.extra_dim

save_folder = os.path.join(".", "visualizations", args.model_name)
os.makedirs(save_folder, exist_ok=True)

fig, ax = plot_extra_dims(extra, color_labels=n_stabilizers)
if fig:
    fig.savefig(os.path.join(save_folder, 'invariant.png'))

for i in range(10):
    print(f"Plotting example {i}")
    fig, axes = plot_images_distributions(mean[i], std[i], mean_next[i], std_next[i], npimages[i], npimages_next[i], N)
    fig.savefig(os.path.join(save_folder, f"test_{i}.png"), bbox_inches='tight')
    plt.close("all")


fig, axes = plot_embeddings_eval(mean_eval, std_eval, N, stabilizers[0])
fig.savefig(os.path.join(save_folder, f"eval_embeddings.png"), bbox_inches='tight')



