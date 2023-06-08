import os
from typing import List, Tuple

import numpy as np
import torch
import torchvision
from dataset_generation.modelnet_regular import get_initial_orientation, shuffle_rows, get_object_ids, OBJECT_DICT_INT, \
    stabilizer_dict, ID_MULTIPLIER
from torch.utils.data import Dataset


class ModelNetDataset(Dataset):
    def __init__(self, render_folder, split, object_type_list, examples_per_object: int, use_random_initial: bool,
                 total_views: int,
                 fixed_number_views: int, shuffle_available_views: bool, use_random_choice: bool, seed: int = 1789,
                 resolution: int = 64):
        self.rng = np.random.default_rng(seed=seed)
        self.render_folder = render_folder
        self.examples_per_object = examples_per_object
        self.use_random_choice = use_random_choice
        self.split = split  # Either train or test
        self.total_views = total_views  # Total number of views available per object
        self.resolution = resolution  # Resolution of the images
        self.object_type_list = object_type_list

        # Properties that depend on the previous ones
        # Join all the object types to be used
        self.object_ids = self.get_object_ids()
        # Get the directories path for each object and the integer class
        self.object_dirs, self.object_type_list = self.get_object_dirs()

        # Get the initial orientations for each object
        self.initial_orientations = get_initial_orientation(use_random_initial, len(self.object_ids), total_views,
                                                            self.rng)
        # Get the available views for each object
        self.orientation_steps = np.arange(0, total_views, total_views // fixed_number_views)
        # Define the available views per object
        self.available_views = (self.initial_orientations[:, None] + self.orientation_steps[None, :]) % total_views

        # Shuffle the views per object for selection
        if shuffle_available_views:
            shuffle_rows(self.available_views, self.rng)

        self.data_list = self.get_data_list()
        self.transforms = self.get_torch_transforms()

    def get_torch_transforms(self):
        if self.resolution == 224:
            transforms = torch.nn.Sequential(
            )
        else:
            transforms = torch.nn.Sequential(
                torchvision.transforms.Resize((self.resolution, self.resolution), antialias=True),
            )
        return transforms


    def get_object_ids(self):
        object_ids = []
        for object_type in self.object_type_list:
            object_ids += get_object_ids(self.render_folder, self.split, object_type)
        return np.array(object_ids)

    @property
    def num_objects(self) -> int:
        return len(self.object_dirs)

    @property
    def num_views(self) -> int:
        return len(self.orientation_steps)

    @property
    def total_pairs(self):
        if self.use_random_choice:
            return self.examples_per_object
        else:
            return self.examples_per_object // 2

    def get_object_dirs(self) -> Tuple[np.array, np.array]:
        object_dirs = []
        object_types_int = []
        for object_id in self.object_ids:
            object_type = "_".join(object_id.split("_")[:-1])
            object_types_int.append(OBJECT_DICT_INT[object_type])
            object_dirs.append(os.path.join(self.render_folder, self.split, object_type, "renders", object_id))
        object_dirs = np.array(object_dirs)
        object_types_int = np.array(object_types_int)
        return object_dirs, object_types_int

    def get_data_list(self) -> List[torch.Tensor]:
        """
        Get the data tuples for the dataset (path1, path2, angle) where the output is the path to the image 1, path to
        the image 2 and the angle between them
        :return:
        """
        angles = []
        path1 = []
        path2 = []
        for num_object, object_dir in enumerate(self.object_dirs):
            object_render_dir = os.path.join(object_dir, os.path.basename(object_dir))
            if self.use_random_choice:
                index1 = self.rng.choice(self.available_views[num_object], size=self.examples_per_object, )
                index2 = self.rng.choice(self.available_views[num_object], size=self.examples_per_object, )
            else:
                index1 = self.available_views[num_object][::2]
                index2 = self.available_views[num_object][1::2]
            # Create the paths for the images
            path1 += list(map(lambda x: object_render_dir + "_" + str(x) + ".png", index1))
            path2 += list(map(lambda x: object_render_dir + "_" + str(x) + ".png", index2))
            # Create the indices for the images
            angles += list((index2 - index1) * 2 * np.pi / self.total_views)

        angles = torch.tensor(np.array(angles)[:, None]).float()
        path1 = np.array(path1)
        path2 = np.array(path2)
        return [path1, path2, angles]

    def __len__(self):
        return len(self.data_list[0])

    def __getitem__(self, idx):
        path1 = self.data_list[0][idx]
        path2 = self.data_list[1][idx]
        action = self.data_list[2][idx]
        image1 = torchvision.io.read_image(path1, torchvision.io.ImageReadMode.RGB)
        image2 = torchvision.io.read_image(path2, torchvision.io.ImageReadMode.RGB)
        image1 = self.transforms(image1).float() / 255.
        image2 = self.transforms(image2).float() / 255.
        return image1, image2, action


class ModelNetDatasetComplete(ModelNetDataset):
    """
    Dataset for the complete modelnet dataset which outputs the image pairs, action, stabilizers and object type int
    """

    def __init__(self, render_folder, split, object_type_list, examples_per_object: int, use_random_initial: bool,
                 total_views: int,
                 fixed_number_views: int, shuffle_available_views: bool, use_random_choice: bool, seed: int = 1789,
                 resolution: int = 64):
        super().__init__(render_folder, split, object_type_list, examples_per_object, use_random_initial, total_views,
                         fixed_number_views, shuffle_available_views, use_random_choice, seed, resolution)

    def get_data_list(self) -> List[torch.Tensor]:
        """
        Get the data tuples for the dataset (path1, path2, angle) where the output is the path to the image 1, path to
        the image 2 and the angle between them
        :return:
        """
        angles = []
        object_types_int = []
        stabilizers = []
        path1 = []
        path2 = []
        orbit_int = []
        for num_object, object_dir in enumerate(self.object_dirs):
            object_id = os.path.basename(object_dir)
            object_type = "_".join(object_id.split("_")[:-1])
            object_render_dir = os.path.join(object_dir, object_id)
            if self.use_random_choice:
                index1 = self.rng.choice(self.available_views[num_object], size=self.examples_per_object, )
                index2 = self.rng.choice(self.available_views[num_object], size=self.examples_per_object, )
            else:
                index1 = self.available_views[num_object][::2]
                index2 = self.available_views[num_object][1::2]
            # Create the paths for the images
            path1 += list(map(lambda x: object_render_dir + "_" + str(x) + ".png", index1))
            path2 += list(map(lambda x: object_render_dir + "_" + str(x) + ".png", index2))
            # Create the indices for the images
            angles += list((index2 - index1) * 2 * np.pi / self.total_views)
            object_types_int += [OBJECT_DICT_INT[object_type]] * len(index1)
            stabilizers += [stabilizer_dict[object_type]] * len(index1)
            orbit_int += [OBJECT_DICT_INT[object_type] * ID_MULTIPLIER + int(object_id.split("_")[-1])] * len(index1)

        angles = torch.tensor(np.array(angles)[:, None]).float()
        stabilizers = torch.tensor(np.array(stabilizers)).int()
        object_types_int = torch.tensor(np.array(object_types_int)).int()
        orbit_int = torch.tensor(np.array(orbit_int)).int()
        path1 = np.array(path1)
        path2 = np.array(path2)
        return [path1, path2, angles, stabilizers, orbit_int, object_types_int]

    def __getitem__(self, idx):
        path1 = self.data_list[0][idx]
        path2 = self.data_list[1][idx]
        action = self.data_list[2][idx]
        stabilizer = self.data_list[3][idx]
        orbit_int = self.data_list[4][idx]
        object_type = self.data_list[5][idx]

        image1 = torchvision.io.read_image(path1, torchvision.io.ImageReadMode.RGB)
        image2 = torchvision.io.read_image(path2, torchvision.io.ImageReadMode.RGB)
        image1 = self.transforms(image1).float() / 255.
        image2 = self.transforms(image2).float() / 255.
        return image1, image2, action, stabilizer, orbit_int, object_type


class ModelNetUniqueDataset(ModelNetDatasetComplete):
    def __init__(self, render_folder, split, object_type_list, examples_per_object: int, use_random_initial: bool,
                 total_views: int,
                 fixed_number_views: int, use_random_choice: bool, seed: int = 1789,
                 resolution: int = 64,
                 index_unique_factors: int = 0):
        self.index_unique_factors = index_unique_factors
        super().__init__(render_folder, split, object_type_list, examples_per_object, use_random_initial, total_views,
                         fixed_number_views, False, use_random_choice, seed, resolution)
        self.filter_data_based_on_unique_factors_()

    def filter_data_based_on_unique_factors_(self) -> np.array:
        """
        Get the unique factors for the object type
        :param object_type: The object type
        :param num_object: The number of the object
        :return: The unique factors
        """
        selection_identifiers = self.data_list[self.index_unique_factors].clone()
        unique_identifiers = torch.unique(selection_identifiers)

        for num_data, data in enumerate(self.data_list):
            new_data = []
            for unique_identifier in unique_identifiers:
                print("Selecting ", unique_identifier)
                selection = torch.where(selection_identifiers == unique_identifier)
                print("Selected_data", data[selection][0])
                new_data.append(data[selection][0])
            print(type(new_data[0]))
            if isinstance(new_data[0], np.str_):
                self.data_list[num_data] = np.array(new_data)
            else:
                self.data_list[num_data] = torch.tensor(new_data)
            print("New data list shape", self.data_list[num_data].shape)


class ModelNetEvalDataset(ModelNetDataset):
    def __init__(self, render_folder, split, object_type_list, examples_per_object: int, use_random_initial: bool,
                 total_views: int,
                 fixed_number_views: int, use_random_choice: bool, seed: int = 1789,
                 resolution: int = 64):
        super().__init__(render_folder, split, object_type_list, examples_per_object, use_random_initial, total_views,
                         fixed_number_views, False, use_random_choice, seed, resolution)

    def get_data_list(self) -> List[torch.Tensor]:
        """
        Get the data tuples for the dataset (path1, path2, angle) where the output is the path to the image 1, path to
        the image 2 and the angle between them
        :return:
        """
        angles = []
        path = []
        object_types_int = []
        stabilizers = []
        orbit_int = []

        for num_object, object_dir in enumerate(self.object_dirs):
            object_id = os.path.basename(object_dir)
            object_type = "_".join(object_id.split("_")[:-1])
            object_render_dir = os.path.join(object_dir, object_id)
            index = self.available_views[num_object]
            # Create the paths for the images
            path += list(map(lambda x: object_render_dir + "_" + str(x) + ".png", index))
            object_types_int += [OBJECT_DICT_INT[object_type]] * len(index)
            stabilizers += [stabilizer_dict[object_type]] * len(index)
            orbit_int += [OBJECT_DICT_INT[object_type] * ID_MULTIPLIER + int(object_id.split("_")[-1])] * len(index)
            # Create the indices for the images
            angles += list(index * 2 * np.pi / self.total_views)

        path = np.array(path)
        angles = torch.tensor(np.array(angles)[:, None]).float()
        stabilizers = torch.tensor(np.array(stabilizers)).int()
        orbit_int = torch.tensor(np.array(orbit_int)).int()
        object_types_int = torch.tensor(object_types_int).int()
        print("Number of examples: ", len(path))
        print("Number of objects: ", len(self.object_dirs))
        print("Number of angles", len(angles))

        return [path, angles, stabilizers, orbit_int, object_types_int]

    def __getitem__(self, idx):
        path = self.data_list[0][idx]
        action = self.data_list[1][idx]
        stabilizer = self.data_list[2][idx]
        orbit = self.data_list[3][idx]
        object_type = self.data_list[4][idx]
        image = torchvision.io.read_image(path, torchvision.io.ImageReadMode.RGB)
        image = self.transforms(image).float() / 255.
        return image, action, stabilizer, orbit, object_type
