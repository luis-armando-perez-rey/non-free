import argparse
import pickle
import numpy as np
import os, sys
import torch
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from utils.dataset_utils import get_dataset
from utils.nn_utils import get_rotated_mean
from utils.plotting_utils import plot_extra_dims, plot_images_distributions, \
    plot_mixture_neurreps, add_image_to_ax, add_distribution_to_ax_torus, save_embeddings_on_circle, yiq_embedding, \
    plot_embeddings_eval_torus
from utils.plotting_utils_so3 import visualize_so3_probabilities
from utils.torch_utils import torch_data_to_numpy


CWD = os.getcwd()
PROJECT_PATH = os.path.dirname(CWD)
QUESSARD_PATH = os.path.join(PROJECT_PATH, "learning-group-structure")

sys.path.append(QUESSARD_PATH)

print("Appended ", PROJECT_PATH)
print("Appended ", QUESSARD_PATH)


from src import data_environment, environments
from added_modules.plotting import plotting
import torch.nn.functional as F
from utils.nn_utils import hitRate_generic

parser = argparse.ArgumentParser()
parser.add_argument('--save-folder', type=str, default='checkpoints', help='Path to saved model')
parser.add_argument('--dataset', type=str, default='dataset', help='Dataset')
parser.add_argument('--dataset_name', nargs="+", type=str, default=['4'], help='Dataset name')
args_eval = parser.parse_args()

model_dir = os.path.join(".", "saved_models", args_eval.save_folder)
encoder_file = os.path.join(model_dir, 'model_encoder.pt')
decoder_file = os.path.join(model_dir, 'model_decoder.pt')
latenv_file = os.path.join(model_dir, 'model_latenv.pt')
meta_file = os.path.join(model_dir, 'metadata.pkl')
args = pickle.load(open(meta_file, 'rb'))['args']
device = torch.device("cuda:" + args.gpu if torch.cuda.is_available() else "cpu")
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
numpy_filepath = os.path.join(f'{args.data_dir}/{args.dataset}/', args.dataset_name[0] + '_data.npy')
images = np.load(numpy_filepath)
obs_env = data_environment.ObjectImageWorld(images)



# dset, eval_dset = get_dataset(args.data_dir, args.dataset, args.dataset_name)
# train_loader = torch.utils.data.DataLoader(dset, batch_size=100, shuffle=True)
# endregion

# region GET MODEL
class Normalize(torch.nn.Module):
    def forward(self, x):
        return F.normalize(x).squeeze()


class Sigmoid(torch.nn.Module):
    def forward(self, x):
        return torch.sigmoid(x).squeeze()

class Unsqueezed(torch.nn.Module):
    def forward(self, x):
        return x.unsqueeze(0)
print("Loading model from: ", encoder_file)
device = 'cpu'
encoder = torch.load(encoder_file).to(device)
decoder = torch.load(decoder_file).to(device)
lat_env = torch.load(decoder_file).to(device)

encoder.eval()

# endregion




# Generate pairs
num_eval_pairs = 10
pairs = []
for _ in range(num_eval_pairs):
    random_object = np.random.randint(0, len(images))
    object_images = images[random_object]
    indexes = np.arange(len(object_images))
    random_indexes = np.random.choice(indexes, 2 * num_eval_pairs)
    pairs_object = np.array([(object_images[i], object_images[j]) for i, j in zip(random_indexes[::2], random_indexes[1::2])])
    pairs.append(pairs_object)
pairs = np.concatenate(pairs, axis=0)
# Permute third with last dimension in pairs
pairs = torch.tensor(np.transpose(pairs, (0, 1, 4, 2, 3))).float().to(device)
print(pairs)

encoded_images = []
encoded_images_next = []
for pair in pairs:
    encoded_images.append(encoder(pair[0]))
    encoded_images_next.append(encoder(pair[1]))
encoded_images = torch.stack(encoded_images)
encoded_images_next = torch.stack(encoded_images_next)
print(encoded_images.shape)
dist_matrix = ((encoded_images.unsqueeze(0) - encoded_images_next.unsqueeze(1))**2).sum(-1)
hitrate = hitRate_generic(dist_matrix, encoded_images.shape[0])
print(hitrate)
# extra_matrix = ((extra.unsqueeze(0) - extra_next.unsqueeze(1)) ** 2).sum(-1)
# hitrate = hitRate_generic(chamfer_matrix + extra_matrix, image.shape[0])
# mu_hit_rate += hitrate.item()

# encoded_image_transformed_flat = encoded_image_transformed.view((batch_size,-1))
# encoded_image_next_flat = encoded_image_next.view((batch_size,-1))
# dist_matrix = ((encoded_image_transformed_flat.unsqueeze(0) - encoded_image_next_flat.unsqueeze(1))**2).sum(-1)
# hitrate = hitRate_generic(dist_matrix, batch_size)
# mu_hitrate += hitrate.item()


if run is not None:
    run.stop()