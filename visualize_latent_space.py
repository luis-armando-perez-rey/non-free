from multiprocessing.sharedctypes import Value
from tkinter import E
from comet_ml import Experiment

from sklearn.decomposition import PCA
import torch
import torch.utils.data
from torch import nn, optim, save, load
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torchvision import transforms
import numpy as np
from scipy.spatial.transform import Rotation as R
import os
import matplotlib.pyplot as plt
import argparse
import pickle
from matplotlib.patches import Ellipse

from models.models_nn import *
from utils.nn_utils import *
from utils.parse_args import get_args
from models.resnet import *

# Import datasets
from datasets.equiv_dset import *

from matplotlib import cm
import ipdb


parser = argparse.ArgumentParser()
parser.add_argument('--save-folder', type=str, default='checkpoints', help='Path to saved model')
args_eval = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#torch.manual_seed(42)
torch.cuda.empty_cache()

model_file = os.path.join(args_eval.save_folder, 'model.pt')
meta_file = os.path.join(args_eval.save_folder, 'metadata.pkl')
args = pickle.load(open(meta_file, 'rb'))['args']



print(args.dataset)
if args.dataset == 'rot-square':
    dset = EquivDataset(f'{args.data_dir}/rot_square/')
else:
    print("Invalid dataset")


train_loader = torch.utils.data.DataLoader(dset,
                                           batch_size=100, shuffle=True)
device = 'cpu'
model = load(model_file).to(device)
model.eval()


img, img_next, action = next(iter(train_loader))
img_shape = img.shape[1:]
npimages = np.transpose(img.detach().cpu().numpy(), axes=[0,2,3,1])
npimages_next = np.transpose(img_next.detach().cpu().numpy(), axes=[0,2,3,1])

mean, logvar, extra = model(img.to(device))
mean_next, logvar_next, extra_next = model(img_next.to(device))

logvar = -4.6*torch.ones(logvar.shape).to(logvar.device)
logvar_next = -4.6*torch.ones(logvar.shape).to(logvar.device)


s = torch.sin(action.squeeze(-1))
c = torch.cos(action.squeeze(-1))
rot = torch.stack([torch.stack([c, -s]), torch.stack([s, c])]).permute((2,0,1)).unsqueeze(1)
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

if extra_dim > 0:
    plt.figure()
    plt.title('First two orbit dimensions')
    plt.scatter(extra[:,0], extra[:,1])
    plt.show()

plt.figure()
for i in range(100):
    print('action: ', 360*action[i] / (2*pi) )
    print(mean[i,0])

    plt.subplot(2,2,1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.gca().set_xlim(0,1)
    plt.gca().set_ylim(0,1)
    plt.title('first')
    plt.imshow(npimages[i], interpolation='nearest', extent=(0,1,0,1))


    plt.subplot(2,2,2)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.gca().set_xlim(0,1)
    plt.gca().set_ylim(0,1)
    plt.title('second')
    plt.imshow(npimages_next[i], interpolation='nearest', extent=(0,1,0,1))

    plt.subplot(2,2,3)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.gca().set_xlim(-1,1)
    plt.gca().set_ylim(-1,1)
    plt.title('before rot')
    for j in range(N):
        ellipse_j = Ellipse(xy=(mean[i,j,0],mean[i,j,1]), width=std[i,j,0], height=std[i,j,1], color='black', linewidth=15, alpha=0.8)
        plt.gca().add_artist(ellipse_j)

    plt.subplot(2,2,4)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.gca().set_xlim(-1,1)
    plt.gca().set_ylim(-1,1)
    plt.title('after rot')
    for j in range(N):
        ellipse_j = Ellipse(xy=(mean_next[i,j,0],mean_next[i,j,1]), width=std_next[i,j,0], height=std_next[i,j,1], color='black', linewidth=15, alpha=0.8)
        plt.gca().add_artist(ellipse_j)



    plt.show()
