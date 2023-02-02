import os
import sys
import numpy as np
import time
import matplotlib.pyplot as plt
from IPython import display
import torch
import random
import neptune

import torch.nn.functional as F
import torch.optim as optim
import matplotlib.gridspec as gridspec

# os.chdir('/home/luis/learning-group-structure/src/flatland/flat_game')
CWD = os.getcwd()
PROJECT_PATH = os.path.dirname(CWD)
QUESSARD_PATH = os.path.join(PROJECT_PATH, "learning-group-structure")

sys.path.append(QUESSARD_PATH)

print("Appended ", PROJECT_PATH)
print("Appended ", QUESSARD_PATH)

# # Imports from Caselles-Dupre forked code
# sys.path.append(os.path.join(CASELLES_PATH, "src"))
#
#
# sys.path.append(os.path.join(CASELLES_PATH, "added_modules"))
# from architectures import dis_lib, teapot, vgg
# from plotting import plotting
#
# # Imports from our project
# import data.data_loader
#
# from modules.general_metric import general_metric
# from experiments import neptune_config
# from modules.utils.plotting import yiq_embedding, plot_latent_dimension_combinations


from utils.parse_args import get_args
# Imported from Quessards project
from src import data_environment, environments
from added_modules.plotting import plotting
from models.resnet import ResNet18Enc, ResNet18Dec
import pickle

# region PARSE ARGUMENTS
parser = get_args()
args = parser.parse_args()
if args.neptune_user != "None":
    from utils.neptune_utils import initialize_neptune_run, save_sys_id

    run = initialize_neptune_run(args.neptune_user, "non-free")
    run["parameters"] = vars(args)
else:
    run = None
print(args)
# endregion

# region PATHS
# Paths were the models will be saved
model_path = os.path.join(args.checkpoints_dir, args.model_name)
os.makedirs(model_path, exist_ok=True)
decoder_file = os.path.join(model_path, "decoder.pt")
# Paths for saving the images
figures_dir = os.path.join(model_path, 'figures')
os.makedirs(figures_dir, exist_ok=True)
model_file_encoder = os.path.join(model_path, 'model_encoder.pt')
model_file_decoder = os.path.join(model_path, "model_decoder.pt")
model_file_latenv = os.path.join(model_path, "model_latenv.pt")
meta_file = os.path.join(model_path, 'metadata.pkl')
if run is not None:
    neptune_id_path = os.path.join(model_path, 'neptune.txt')
    save_sys_id(run, neptune_id_path)

# endregion

# region SAVE METADATA
# Save the arguments
pickle.dump({'args': args}, open(meta_file, 'wb'))
# endregion

# region TORCH SETUP
# Print set up torch device, empty cache and set random seed
torch.cuda.empty_cache()
# torch.manual_seed(args.seed)
device = torch.device("cuda:" + args.gpu if torch.cuda.is_available() else "cpu")
print('Using device: ', device)
# endregion


# from env import Env
n_sgd_steps = 3000 * 4
n_sgd_steps = 1000
n_sgd_steps = 100
n_sgd_steps = args.epochs
ep_steps = 5
batch_eps = 16
entanglement_target = 0
z_dim = args.latent_dim
architecture = "dis_lib"

# noinspection PyArgumentList
params = torch.FloatTensor([1, 1, 0.5, 0, 0])


def calc_entanglement(params):
    params = params.abs().pow(2)
    return params.sum() - params.max()




calc_entanglement(params)

numpy_filepath = os.path.join(f'{args.data_dir}/{args.dataset}/', args.dataset_name[0] + '_data.npy')
images = np.load(numpy_filepath)
obs_env = data_environment.ObjectImageWorld(images)

print("LOADING DATASET {}".format(args.dataset_name), "Images shape",  images.shape)


# Load images and create data enviroment


# # Save example images from the dataset
# plt.figure(figsize=(3, 3))
# gs1 = gridspec.GridSpec(3, 3)
# gs1.update(wspace=0.02, hspace=0.02)
# plt.grid(None)
# state = obs_env.reset()
# for i in range(9):
#     ax = plt.subplot(gs1[i])
#     ax.axis('off')
#     ax.set_aspect('equal')
#     if state.shape[-1] == 1:
#         ax.imshow(state[:, :, 0])
#     else:
#         ax.imshow(state)
#     display.display(plt.gcf())
#     time.sleep(0.2)
#     display.clear_output(wait=True)
#     action = random.sample([0, 1, 2, 3], k=1)[0]
#     #     action = 2
#     # print(env.env.agent.body.position)
#     state = obs_env.step(action)
# plt.savefig(dataset_parameters["data"] + "_env.png", bbox_inches='tight')
#
# obs = obs_env.reset().permute(-1, 0, 1).float()
# image_size = obs.shape[1:]
#
# # Neptune Experiment
# group = "TUe"
# api_token = neptune_config.API_KEY  # read api token from neptune config file
# upload_source_files = ["caselles_code.py"]  # OPTIONAL: save the source code used for the experiment
# neptune.init(project_qualified_name=group + "/sandbox", api_token=api_token)
#
class Normalize(torch.nn.Module):
    def forward(self, x):
        return F.normalize(x).squeeze()


class Sigmoid(torch.nn.Module):
    def forward(self, x):
        return torch.sigmoid(x).squeeze()

class Unsqueezed(torch.nn.Module):
    def forward(self, x):
        return x.unsqueeze(0)



# DEFINE NETWORK FOR TRAINING

lat_env = environments.LatentWorld(dim=z_dim,
                                   n_actions=obs_env.action_space.n, device=device)

# Define encoder and decoder as a sequential model
encoder = torch.nn.Sequential(Unsqueezed(), ResNet18Enc(z_dim=args.latent_dim), Normalize())
decoder = torch.nn.Sequential(Unsqueezed(), ResNet18Dec(z_dim=args.latent_dim), Sigmoid())

encoder.to(device)
decoder.to(device)

# noinspection PyUnresolvedReferences
optimizer_dec = optim.Adam(decoder.parameters(),
                           lr=1e-2,
                           weight_decay=0)

# noinspection PyUnresolvedReferences
optimizer_enc = optim.Adam(encoder.parameters(),
                           lr=1e-2,
                           weight_decay=0)

optimizer_rep = optim.Adam(lat_env.get_representation_params(),
                           lr=1e-2,
                           weight_decay=0)

losses = []
entanglement = []


# START TRAINING
i = 0

t_start = time.time()

temp = 0
print("START TRAINING")


while i < n_sgd_steps:

    loss = torch.zeros(1).to(device)
    reconstruction_error = torch.zeros(1).to(device)

    for _ in range(batch_eps):
        t_ep = -1
        while t_ep < ep_steps:
            if t_ep == -1:
                obs_x = obs_env.reset().permute(-1, 0, 1).float().to(device)
                encoded_image = encoder(obs_x)
                obs_z = lat_env.reset(encoded_image)
            else:
                action = obs_env.action_space.sample().item()
                obs_x = obs_env.step(action).permute(-1, 0, 1).float().to(device)
                obs_z = lat_env.step(action)

            t_ep += 1
            obs_x_recon = decoder(obs_z.to(device)).to(device)
            reconstruction_error += F.binary_cross_entropy(obs_x_recon, obs_x)
            loss += reconstruction_error

    loss /= (ep_steps * batch_eps)
    reconstruction_error /= (ep_steps * batch_eps)
    raw_loss = loss.item()

    reg_loss = sum([calc_entanglement(r.thetas) for r in lat_env.action_reps]) / 4

    loss += (reg_loss - entanglement_target).abs() * 1e-2

    # log complete loss
    if run is not None:
        run['entanglement_loss'].log(reg_loss)
        run['batch_loss'].log(loss)
        run['reconstruction_error'].log(reconstruction_error)

    losses.append(raw_loss)
    entanglement.append(reg_loss.item())

    optimizer_dec.zero_grad()
    optimizer_enc.zero_grad()
    optimizer_rep.zero_grad()
    loss.to(device)
    loss.backward()
    optimizer_enc.step()
    optimizer_dec.step()
    optimizer_rep.step()

    # Remember to clear the cached action representations after we update the parameters!
    lat_env.clear_representations()

    i += 1

    if i % 10 == 0:
        print("iter {} : loss={:.3f} : entanglement={:.2e} : last 10 iters in {:.3f}s".format(
            i, raw_loss, reg_loss.item(), time.time() - t_start
        ), end="\r" if i % 100 else "\n")
        t_start = time.time()




# Save model
torch.save(encoder, model_file_encoder)
torch.save(decoder, model_file_decoder)
torch.save(lat_env, model_file_latenv)


# Get plots
fig, _ = plotting.plot_action_distribution(lat_env)
if run is not None:
    run[f"action_dist"].upload(fig)
fig, _ = plotting.plot_reconstructions(obs_env, encoder, decoder, 1, device = device)
if run is not None:
    run[f"reconstruction"].upload(fig)


# print("START CREATING EMBEDDINGS")
# latent_embeddings = produce_embeddings(images_dataset, encoder, device = device)
# angles = []
# # num_factors = 64
# embeddings_flat = latent_embeddings.reshape((np.product(images_dataset.images.shape[:2]), z_dim))
# for factor in range(2):
#     angles.append(2 * np.pi * np.array(range(images_dataset.images.shape[factor])) / images_dataset.images.shape[factor])
# factor_meshes = np.meshgrid(*angles, indexing="ij")
# factor_mesh = np.stack(factor_meshes, axis=-1)  # (n1, n2, n3, ..., n_n_factors ,n_factors)
# flat_factor_mesh = factor_mesh.reshape((np.product(images_dataset.images.shape[:2]), 2))
# colors_flat = yiq_embedding(flat_factor_mesh[:, 0], flat_factor_mesh[:, 1])
# fig, axes = plot_latent_dimension_combinations(embeddings_flat, colors_flat)
# neptune.log_image("plots", fig, image_name="embeddings")
# np.save(save_path, latent_embeddings)


