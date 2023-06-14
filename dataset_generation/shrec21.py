import os
from typing import List, Tuple

import numpy as np
import torch
import torchvision
from dataset_generation.modelnet_regular import get_initial_orientation, shuffle_rows, get_object_ids, OBJECT_DICT_INT, \
    stabilizer_dict, ID_MULTIPLIER
from torch.utils.data import Dataset
from dataset_generation.modelnet_regular import get_object_ids


def separate_class_details(line):
    str_class, _, num_objects = line.split(" ")
    return str_class, int(num_objects)


def process_cla_file(file_path):
    """
    Process the .cla file and return a dictionary with the following structure:
    {class_id: {"obj_ids": [obj_id1, obj_id2, ...], "class_str": "class_name", "num_objects": num_objects}}
    :param file_path:
    :return:
    """
    with open(file_path) as file:
        lines = [line.strip() for line in file.readlines()]
        num_classes, num_objects = [int(value) for value in lines[1].split(" ")]
        class_index = 3
        data_dictionary = {}
        for int_class in range(num_classes):
            object_class_dictionary = {}
            str_class, num_objects = separate_class_details(lines[class_index])
            object_class_dictionary["obj_ids"] = [int(obj_id) for obj_id in
                                                  lines[class_index + 1: class_index + num_objects + 1]]
            object_class_dictionary["class_str"] = str_class
            object_class_dictionary["num_objects"] = num_objects
            data_dictionary[int_class] = object_class_dictionary
            class_index += num_objects + 2
    return data_dictionary


AVAILABLE_OBJECTS = ["basin", "bowl", "figurine", "jar", "pitcher", "plate", "pot", "vase"]
OBJECT_DICT_INT = {object_type: i for i, object_type in enumerate(AVAILABLE_OBJECTS)}


class PeruvianObjects(Dataset):
    def __init__(self, render_folder, split, object_type_list, examples_per_object: int, use_random_initial: bool,
                 total_views: int,
                 fixed_number_views: int, shuffle_available_views: bool, use_random_choice: bool,
                 seed: int = 1789,
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
        if self.resolution == 224:
            self.transforms = torch.nn.Sequential(
            )
        self.transforms = torch.nn.Sequential(
            torchvision.transforms.Resize((self.resolution, self.resolution), antialias=True),
        )



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
