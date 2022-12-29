import torch
from torch.utils.data import DataLoader
import numpy as np
import os
from typing import List, Optional
import pickle
import scipy

class EquivDataset(torch.utils.data.Dataset):
    def __init__(self, path: str, list_dataset_names=None, greyscale: bool = False,
                 max_data_per_dataset: int = -1, so3_matrices: bool = False):
        if list_dataset_names is None:
            list_dataset_names = ["equiv"]
        print(f"Loading the datasets {list_dataset_names}")
        data = np.load(path + list_dataset_names[0] + '_data.npy', mmap_mode='r+')

        lbls = np.load(path + list_dataset_names[0] + '_lbls.npy', mmap_mode='r+')
        if max_data_per_dataset > 0:
            num_selected_data = np.min([max_data_per_dataset, len(data)])
            if max_data_per_dataset > len(data):
                print(
                    f"Warning: max_data_per_dataset is larger than the number of data in the dataset {list_dataset_names[0]}")
            data = data[:num_selected_data]
            lbls = lbls[:num_selected_data]
        self.data = data
        self.lbls = lbls

        for dataset_name in list_dataset_names[1:]:
            data = np.load(path + dataset_name + '_data.npy', mmap_mode='r+')
            lbls = np.load(path + dataset_name + '_lbls.npy', mmap_mode='r+')
            if max_data_per_dataset > 0:
                num_selected_data = np.min([max_data_per_dataset, len(data)])
                if max_data_per_dataset > len(data):
                    print(
                        f"Warning: max_data_per_dataset is larger than the number of data in the dataset {dataset_name}")
                data = data[:num_selected_data]
                lbls = lbls[:num_selected_data]
            self.data = np.concatenate([self.data, data], axis=0)
            self.lbls = np.concatenate([self.lbls, lbls], axis=0)
        if so3_matrices:
            self.lbls = self.angles2so3(self.lbls)
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
    @staticmethod
    def angles2so3(angles):
        """
        Converts a vector of angles to a rotation matrix about the z axis
        :param angles: vector of angles in radians
        :return: rotation matrix
        """
        # Initialize rotation matrix
        rotation_matrix = np.zeros((angles.shape[0], 3, 3), dtype = angles.dtype)
        # Fill out matrix entries
        cos_angle = np.cos(angles)
        sin_angle = np.sin(angles)
        rotation_matrix[:, 0, 0] = cos_angle
        rotation_matrix[:, 0, 1] = -sin_angle
        rotation_matrix[:, 1, 0] = sin_angle
        rotation_matrix[:, 1, 1] = cos_angle
        rotation_matrix[:, 2, 2] = 1.
        return rotation_matrix

    @property
    def flat_images(self):
        return self.data.reshape(-1, *self.data.shape[2:])

    @property
    def flat_images_numpy(self):
        return np.transpose(self.flat_images, axes=(0, 2, 3, 1))

    @property
    def flat_stabs(self):
        if len(self.stabs.shape) == 3:
            return self.stabs.reshape(-1, self.stabs.shape[-1])
        else:
            return self.stabs.reshape((-1))

    @property
    def flat_lbls(self):
        if self.lbls is not None:
            if len(self.lbls.shape) == 3:
                return self.lbls.reshape(-1, self.lbls.shape[-1])
            else:
                return self.lbls.reshape((-1))
        else:
            return None

    @property
    def num_objects(self):
        return self.data.shape[0]


class EquivDatasetStabs(EquivDataset):
    """
    Equivariance dataset that produces the same output as EquivDataset, but with the addition of the cardinality of the
    stabilizers corresponding to the image pair.
    """

    def __init__(self, path: str, list_dataset_names: List[str] = ["equiv"], greyscale: bool = False,
                 max_data_per_dataset: int = -1, so3_matrices: bool = False):
        super().__init__(path, list_dataset_names, greyscale, max_data_per_dataset, so3_matrices)
        for dataset_name in list_dataset_names:
            assert os.path.exists(
                path + dataset_name + '_stabilizers.npy'), f"{dataset_name}_stabilizers.npy cardinality file not found"
        stabs = np.load(path + list_dataset_names[0] + '_stabilizers.npy', mmap_mode='r+')
        if max_data_per_dataset > 0:
            num_selected_data = np.min([max_data_per_dataset, len(stabs)])
            if max_data_per_dataset > len(stabs):
                print(
                    f"Warning: max_data_per_dataset is larger than the number of data in the dataset {list_dataset_names[0]}")
            stabs = stabs[:num_selected_data]
        self.stabs = stabs
        for dataset_name in list_dataset_names[1:]:
            stabs = np.load(path + dataset_name + '_stabilizers.npy', mmap_mode='r+')
            if max_data_per_dataset > 0:
                num_selected_data = np.min([max_data_per_dataset, len(stabs)])
                if max_data_per_dataset > len(stabs):
                    print(
                        f"Warning: max_data_per_dataset is larger than the number of data in the dataset {dataset_name}")
                stabs = stabs[:num_selected_data]
            self.stabs = np.concatenate([self.stabs, stabs], axis=0)

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
            self.stabs = np.concatenate(
                [self.stabs, np.load(path + dataset_name + '_eval_stabilizers.npy', mmap_mode='r+')],
                axis=0)
            if self.lbls is not None:
                self.lbls = np.concatenate([self.lbls, np.load(path + dataset_name + '_eval_lbls.npy', mmap_mode='r+')],
                                           axis=0)

    @property
    def flat_images(self):
        return self.data.reshape(-1, *self.data.shape[2:])

    @property
    def flat_images_numpy(self):
        return np.transpose(self.flat_images, axes=(0, 2, 3, 1))

    @property
    def flat_stabs(self):
        if len(self.stabs.shape) == 3:
            return self.stabs.reshape(-1, self.stabs.shape[-1])
        else:
            return self.stabs.reshape((-1))

    @property
    def flat_lbls(self):
        if self.lbls is not None:
            if len(self.lbls.shape) == 3:
                return self.lbls.reshape(-1, self.lbls.shape[-1])
            else:
                return self.lbls.reshape((-1))
        else:
            return None

    @property
    def num_objects(self):
        return self.data.shape[0]


def PlatonicMerged(N, big=True, data_dir='data'):
    pyra = PlatonicDataset('tetra', N=N, big=big, data_dir=data_dir)
    octa = PlatonicDataset('octa', N=N, big=big, data_dir=data_dir)
    cube = PlatonicDataset('cube', N=N, big=big, data_dir=data_dir)
    return torch.utils.data.ConcatDataset([cube, octa, pyra])


class PlatonicDataset(torch.utils.data.Dataset):
    def __init__(self, platonic, N, big=True, width=64, data_dir='data', logarithmic=False):
        self.classes = {'cube': 0, 'tetra': 1, 'octa': 2}
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

        action = torch.from_numpy(np.expand_dims(action, axis=0)).float()

        return img1, img2, action  # torch.Tensor([self.classes[self.platonic]]).long()

    def __len__(self):
        return len(self.data)
