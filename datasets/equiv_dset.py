import torch
from torch.utils.data import DataLoader
import numpy as np
import os
from typing import List


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
        assert os.path.exists(
            path + list_dataset_names[
                0] + '_stabilizers.npy'), f"{list_dataset_names[0]}_stabilizers.npy cardinality file not found"
        self.stabs = np.load(path + list_dataset_names[0] + '_stabilizers.npy', mmap_mode='r+')
        for dataset_name in list_dataset_names[1:]:
            self.stack = np.concatenate([self.stabs, np.load(path + dataset_name + '_stabilizers.npy', mmap_mode='r+')],
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
    def __init__(self, path: str, list_dataset_names: List[str]):
        self.data = np.load(path + list_dataset_names[0] + '_eval_data.npy', mmap_mode='r+')
        self.stabs = np.load(path + list_dataset_names[0] + '_eval_stabilizers.npy', mmap_mode='r+')
        for dataset_name in list_dataset_names[1:]:
            self.data = np.concatenate([self.data, np.load(path + dataset_name + '_eval_data.npy', mmap_mode='r+')],
                                       axis=0)
            self.stabs = np.concatenate([self.stabs, np.load(path + dataset_name + '_eval_stabilizers.npy', mmap_mode='r+')],
                                       axis=0)

