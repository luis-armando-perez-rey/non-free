import torch
from torch.utils.data import DataLoader
import numpy as np
import os
from typing import List, Optional
import pickle
import scipy


class EquivDataset(torch.utils.data.Dataset):
    def __init__(self, path: str, list_dataset_names: List[str], greyscale: bool = False,
                 max_data_per_dataset: int = -1, so3_matrices: bool = False):
        print(f"Loading the datasets {list_dataset_names}")
        # Properties
        self.max_data_per_dataset = max_data_per_dataset
        self.path = path
        self.greyscale = greyscale
        self.list_dataset_names = list_dataset_names

        # Load the data
        self.data = self.load_data("data")
        self.lbls = self.load_data("lbls")
        if so3_matrices:
            self.lbls = self.angles2so3(self.lbls)

    def load_data(self, factor_name):
        """
        Load the data from the numpy files created during data generation examples: data, lbls, stabs
        :param factor_name: name of the factor data to be loaded
        :return:
        """
        factor_filepath = self.path + self.list_dataset_names[0] + '_' + factor_name + '.npy'
        assert os.path.exists(factor_filepath), "{} file not found".format(factor_filepath)
        total_factors = np.load(factor_filepath, mmap_mode='r+')
        for dataset_name in self.list_dataset_names[1:]:
            # Assert if factor file exists
            factor_filepath = os.path.join(self.path, dataset_name + "_" + factor_name + ".npy")
            assert os.path.exists(factor_filepath), "{} file not found".format(factor_filepath)
            # Load factor file
            factors = np.load(factor_filepath, mmap_mode='r+')
            # Select data based on maximum data per dataset selected
            factors = self.select_data(factors)
            # Concatenate factors
            total_factors = np.concatenate([total_factors, factors], axis=0)
        return total_factors

    def select_data(self, data):
        if self.max_data_per_dataset > 0:
            num_selected_data = np.min([self.max_data_per_dataset, len(data)])
            if self.max_data_per_dataset > len(data):
                print(f"Warning: max_data_per_dataset is larger than the number of data")
            data = data[:num_selected_data]
        return data

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
        rotation_matrix = np.zeros((angles.shape[0], 3, 3), dtype=angles.dtype)
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

    def __init__(self, path: str, list_dataset_names: List[str], greyscale: bool = False,
                 max_data_per_dataset: int = -1, so3_matrices: bool = False):
        super().__init__(path, list_dataset_names, greyscale, max_data_per_dataset, so3_matrices)
        self.stabs = self.load_data("stabilizers")

    def __getitem__(self, index):
        if self.greyscale:
            return torch.FloatTensor(self.data[index, 0]).unsqueeze(0), torch.FloatTensor(
                self.data[index, 1]).unsqueeze(0), torch.FloatTensor((self.lbls[index],)), torch.FloatTensor((
                self.stabs[index],))
        else:
            return torch.FloatTensor(self.data[index, 0]), torch.FloatTensor(self.data[index, 1]), torch.FloatTensor(
                (self.lbls[index],)), torch.FloatTensor((
                self.stabs[index],))


class FactorDataset(EquivDatasetStabs):
    def __init__(self, path: str, list_dataset_names: List[str], greyscale: bool = False,
                 max_data_per_dataset: int = -1, so3_matrices: bool = False, factor_list: List[str] = None):
        super().__init__(path, list_dataset_names, greyscale, max_data_per_dataset, so3_matrices)
        self.factors = []
        for factor_name in factor_list:
            self.factors.append(self.load_data(factor_name))

    def __getitem__(self, index):
        output_factors = [torch.FloatTensor((factors[index],)) for factors in self.factors]
        if self.greyscale:
            return [torch.FloatTensor(self.data[index, 0]).unsqueeze(0),
                    torch.FloatTensor(self.data[index, 1]).unsqueeze(0),
                    torch.FloatTensor((self.lbls[index],)),
                    torch.FloatTensor((self.stabs[index],)),
                    *output_factors]
        else:
            return [torch.FloatTensor(self.data[index, 0]),
                    torch.FloatTensor(self.data[index, 1]),
                    torch.FloatTensor((self.lbls[index],)),
                    torch.FloatTensor((self.stabs[index],)),
                    *output_factors]


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
