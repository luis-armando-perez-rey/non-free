import torch
from torch.utils.data import DataLoader
import numpy as np
import os
from typing import List
import pickle
import scipy


class EquivDataset(torch.utils.data.Dataset):
    def __init__(self, path: str, list_dataset_names: List[str] = ["equiv"], greyscale: bool = False):
        print(f"Loading the datasets {list_dataset_names}")
        self.data = np.load(path + list_dataset_names[0] + '_data.npy', mmap_mode='r+')
        self.lbls = np.load(path + list_dataset_names[0] + '_lbls.npy', mmap_mode='r+')
        for dataset_name in list_dataset_names[1:]:
            self.data = np.concatenate([self.data, np.load(path + dataset_name + '_data.npy', mmap_mode='r+')],
                                       axis=0)
            self.lbls = np.concatenate([self.lbls, np.load(path + dataset_name + '_lbls.npy', mmap_mode='r+')],
                                       axis=0)
        self.greyscale = greyscale

    def __getitem__(self, index):
        if self.greyscale:
            return torch.FloatTensor(self.data[index, 0]).unsqueeze(0), torch.FloatTensor(
                self.data[index, 1]).unsqueeze(0), torch.FloatTensor((self.lbls[index],))
        else:
            return torch.FloatTensor(self.data[index, 0]), torch.FloatTensor(self.data[index, 1]), torch.FloatTensor(
                (self.lbls[index],))

    def __len__(self):
        return len(self.data)


class EquivDatasetStabs(EquivDataset):
    """
    Equivariance dataset that produces the same output as EquivDataset, but with the addition of the cardinality of the
    stabilizers corresponding to the image pair.
    """

    def __init__(self, path: str, list_dataset_names: List[str] = ["equiv"], greyscale: bool = False):
        super().__init__(path, list_dataset_names, greyscale)
        for dataset_name in list_dataset_names:
            assert os.path.exists(
                path + dataset_name + '_stabilizers.npy'), f"{dataset_name}_stabilizers.npy cardinality file not found"
        self.stabs = np.load(path + list_dataset_names[0] + '_stabilizers.npy', mmap_mode='r+')
        for dataset_name in list_dataset_names[1:]:
            self.stabs = np.concatenate([self.stabs, np.load(path + dataset_name + '_stabilizers.npy', mmap_mode='r+')],
                                        axis=0)

    def __getitem__(self, index):
        if self.greyscale:
            return torch.FloatTensor(self.data[index, 0]).unsqueeze(0), torch.FloatTensor(
                self.data[index, 1]).unsqueeze(0), torch.FloatTensor((self.lbls[index],)), torch.FloatTensor((
                self.stabs[index],))
        else:
            return torch.FloatTensor(self.data[index, 0]), torch.FloatTensor(self.data[index, 1]), torch.FloatTensor(
                (self.lbls[index],)), torch.FloatTensor((
                self.stabs[index],))


class EvalDataset(torch.utils.data.Dataset):
    def __init__(self, path: str, list_dataset_names: List[str], load_labels: bool = True):
        self.data = np.load(path + list_dataset_names[0] + '_eval_data.npy', mmap_mode='r+')
        self.stabs = np.load(path + list_dataset_names[0] + '_eval_stabilizers.npy', mmap_mode='r+')
        if os.path.isfile(path + list_dataset_names[0] + '_eval_lbls.npy'):
            self.lbls = np.load(path + list_dataset_names[0] + '_eval_lbls.npy', mmap_mode='r+')
        else:
            self.lbls = None
        for dataset_name in list_dataset_names[1:]:
            self.data = np.concatenate([self.data, np.load(path + dataset_name + '_eval_data.npy', mmap_mode='r+')],
                                       axis=0)
            self.stabs = np.concatenate([self.stabs, np.load(path + dataset_name + '_eval_stabilizers.npy', mmap_mode='r+')],
                                       axis=0)
            if self.lbls is not None:
                self.lbls = np.concatenate([self.lbls, np.load(path + dataset_name + '_eval_lbls.npy', mmap_mode='r+')],
                                           axis=0)


def PlatonicMerged(N, big=True, data_dir='data'):
    pyra = PlatonicDataset('tetra', N=N, big=big, data_dir=data_dir)
    octa = PlatonicDataset('octa', N=N, big=big, data_dir=data_dir)
    cube = PlatonicDataset('cube', N=N, big=big, data_dir=data_dir)
    return torch.utils.data.ConcatDataset([cube, octa, pyra])

class PlatonicDataset(torch.utils.data.Dataset):
    def __init__(self, platonic, N, big=True, width=64, data_dir='data', logarithmic=False):

        self.classes = {'cube':0, 'tetra':1, 'octa':2}
        self.platonic = platonic
        self.logarithmic = logarithmic

        postfix = '-big' if big else ''
        path = f'{data_dir}/platonic/{platonic}_black_data.pkl'
        with open(path, 'rb') as f:
            self.data = pickle.load(f)

    def __getitem__(self, idx):
        img1, img2, action = self.data[idx]

        img1 = torch.from_numpy(img1).float() / 255.
        img2 = torch.from_numpy(img2).float() / 255.

        if self.logarithmic:
            action = scipy.linalg.logm(action)

        action = torch.from_numpy(action).float()

        return img1, img2, action #torch.Tensor([self.classes[self.platonic]]).long()

    def __len__(self):
        return len(self.data)
