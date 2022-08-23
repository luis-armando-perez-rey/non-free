import torch
from torch.utils.data import DataLoader
import numpy as np
import os


class EquivDataset(torch.utils.data.Dataset):
    def __init__(self, path: str, dataset_name: str = "equiv", greyscale: bool = False):
        self.data = np.load(path + dataset_name + '_data.npy', mmap_mode='r+')
        self.lbls = np.load(path + dataset_name + '_lbls.npy', mmap_mode='r+')
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

    def __init__(self, path: str, dataset_name: str = "equiv", greyscale: bool = False):
        super().__init__(path, dataset_name, greyscale)
        assert os.path.exists(
            path + dataset_name + '_stabilizers.npy'), f"{dataset_name}_stabilizers.npy cardinality file not found"
        self.stabs = np.load(path + dataset_name + '_stabilizers.npy', mmap_mode='r+')

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
    def __init__(self, path: str, dataset_name: str):
        self.data = np.load(path + dataset_name + '_eval_data.npy', mmap_mode='r+')
        self.stabs = np.load(path + dataset_name + '_eval_stabilizers.npy', mmap_mode='r+')
