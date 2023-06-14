import os
import numpy as np
import json
import torch
import glob
from torch.utils.data import Dataset
from typing import Tuple, Optional, List
from PIL import Image
import torchvision

AVAILABLE_OBJECTS = ["airplane", "bathtub", "bed", "bench",
                     "bookshelf", "bottle", "bowl", "car",
                     "chair", "cone", "cup", "curtain",
                     "desk", "door", "dresser", "flower_pot",
                     "glass_box", "guitar", "keyboard", "lamp",
                     "laptop", "mantel", "monitor", "night_stand",
                     "person", "piano", "plant", "radio",
                     "range_hood", "sink", "sofa", "stairs",
                     "stool", "table", "tent", "toilet",
                     "tv_stand", "vase", "wardrobe", "xbox",
                     #shrec21
                     "basin", "bowl", "figurine", "jar", "pitcher", "plate", "pot"
                     ]
ID_MULTIPLIER = 10000
OBJECT_DICT_INT = {object_type: i for i, object_type in enumerate(AVAILABLE_OBJECTS)}

stabilizer_dict = {object_type: 1 for object_type in AVAILABLE_OBJECTS}


def process_pil_image(img):
    img = np.array(img)[:, :, :3]
    img = np.moveaxis(img, -1, 0)
    img = img / 255.0
    return img


def get_object_ids(render_folder, split, object_type) -> List[str]:
    object_ids = os.listdir(os.path.join(render_folder, split, object_type, "renders"))
    return object_ids


# Make a pytorch dataloader class

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

        self.data_tuple = self.get_data_tuples()

        self.transforms = torch.nn.Sequential(
            torchvision.transforms.Resize((self.resolution, self.resolution)),
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

    def get_data_tuples(self) -> Tuple[np.array, np.array, np.array]:
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
        return path1, path2, angles

    def __len__(self):
        return len(self.data_tuple[0])

    def __getitem__(self, idx):
        path1 = self.data_tuple[0][idx]
        path2 = self.data_tuple[1][idx]
        action = self.data_tuple[2][idx]
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

    def get_data_tuples(self) -> Tuple[np.array, np.array, torch.Tensor, torch.Tensor, torch.Tensor]:
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

        angles = torch.tensor(np.array(angles)[:, None]).float()
        stabilizers = torch.tensor(np.array(stabilizers)).int()
        object_types_int = torch.tensor(np.array(object_types_int)).int()
        path1 = np.array(path1)
        path2 = np.array(path2)
        return path1, path2, angles, stabilizers, object_types_int

    def __getitem__(self, idx):
        path1 = self.data_tuple[0][idx]
        path2 = self.data_tuple[1][idx]
        action = self.data_tuple[2][idx]
        stabilizer = self.data_tuple[3][idx]
        object_type = self.data_tuple[4][idx]

        image1 = torchvision.io.read_image(path1, torchvision.io.ImageReadMode.RGB)
        image2 = torchvision.io.read_image(path2, torchvision.io.ImageReadMode.RGB)
        image1 = self.transforms(image1).float() / 255.
        image2 = self.transforms(image2).float() / 255.
        return image1, image2, action, stabilizer, object_type


class ModelNetEvalDataset(ModelNetDataset):
    def __init__(self, render_folder, split, object_type_list, examples_per_object: int, use_random_initial: bool,
                 total_views: int,
                 fixed_number_views: int, use_random_choice: bool, seed: int = 1789,
                 resolution: int = 64):
        super().__init__(render_folder, split, object_type_list, examples_per_object, use_random_initial, total_views,
                         fixed_number_views, False, use_random_choice, seed, resolution)

    def get_data_tuples(self) -> Tuple[np.array, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

        return path, angles, stabilizers, orbit_int, object_types_int

    def __getitem__(self, idx):
        path = self.data_tuple[0][idx]
        action = self.data_tuple[1][idx]
        stabilizer = self.data_tuple[2][idx]
        orbit = self.data_tuple[3][idx]
        object_type = self.data_tuple[4][idx]
        image = torchvision.io.read_image(path, torchvision.io.ImageReadMode.RGB)
        image = self.transforms(image).float() / 255.
        return image, action, stabilizer, orbit, object_type


def shuffle_rows(array: np.ndarray, rng: np.random.Generator) -> None:
    """
    Shuffle the rows of a 2D array
    :param array: array to shuffle
    :param rng: random number generator
    :return:
    """
    assert array.ndim == 2, "Array must be 2D"
    for row in array:
        rng.shuffle(row)


def get_initial_orientation(use_random_initial: bool, num_objects: int, total_views: int, rng):
    if use_random_initial:
        random_initial = rng.integers(total_views, size=num_objects)
    else:
        random_initial = np.zeros(num_objects, dtype=int)
    return random_initial


def generate_training_data(render_folder, dataset_folder, dataset_name: str, object_type: str, split: str = "train",
                           examples_per_object: int = 1, use_random_initial: bool = True, fixed_number_views: int = 12,
                           resolution=64, use_random_choice: bool = True, seed: int = 17, total_views: int = 360):
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    rng = np.random.default_rng(seed=seed)
    equiv_data = []
    equiv_lbls = []
    equiv_stabilizers = []
    orbit_info = []
    class_info = []

    object_ids = get_object_ids(render_folder, split, object_type)
    # Define the initial orientation for each object
    initial_orientations = get_initial_orientation(use_random_initial, len(object_ids), total_views, rng)
    # Define the orientation steps for each object
    orientation_steps = np.arange(0, total_views, total_views // fixed_number_views)
    # Define the available views for each object
    available_views = (initial_orientations[:, None] + orientation_steps[None, :]) % total_views
    for num_object, object_id in enumerate(object_ids):
        render_path = os.path.join(render_folder, split, object_type, "renders", object_id)
        id_path = os.path.join("/data", render_folder, split, object_type, "identifiers")
        identifiers = os.listdir(id_path)
        with open(os.path.join(id_path, identifiers[0])) as user_file:
            identifiers = json.load(user_file)

        rng.shuffle(available_views[num_object])
        if use_random_choice:
            total_pairs = examples_per_object
        else:
            total_pairs = examples_per_object // 2
        for num_example in range(total_pairs):
            if use_random_choice:
                index1 = rng.choice(available_views[num_object])
                index2 = rng.choice(available_views[num_object])
            else:
                index1 = available_views[num_object][2 * num_example]
                index2 = available_views[num_object][2 * num_example + 1]
            angle = identifiers["rotations"][index2] - identifiers["rotations"][index1]
            image1_path = os.path.join(render_path, object_id + "_" + str(index1) + ".png")
            image2_path = os.path.join(render_path, object_id + "_" + str(index2) + ".png")
            image1 = Image.open(image1_path)
            image2 = Image.open(image2_path)
            if resolution != 224:
                image1 = image1.resize((resolution, resolution))
                image2 = image2.resize((resolution, resolution))
            image1 = process_pil_image(image1)
            image2 = process_pil_image(image2)

            equiv_data.append([image1, image2])
            equiv_lbls.append(angle)
            print(
                "Generating image with digit number {}, num example {} angle {}".format(object_id, num_example,
                                                                                        angle))
            equiv_stabilizers.append(stabilizer_dict[object_type])
            orbit_info.append([OBJECT_DICT_INT[object_type] * ID_MULTIPLIER + int(object_id.split("_")[-1])] * 2)
            class_info.append([OBJECT_DICT_INT[object_type]] * 2)

    equiv_data = np.array(equiv_data)
    equiv_lbls = np.array(equiv_lbls)
    equiv_stabilizers = np.array(equiv_stabilizers)
    orbit_info = np.array(orbit_info)
    class_info = np.array(class_info)

    print("Equiv data shape", equiv_data.shape)
    print("Equiv lbls shape", equiv_lbls.shape)
    print("Equiv stabilizers shape", equiv_stabilizers.shape)
    print("Orbit info shape", orbit_info.shape)
    print("Class info shape", class_info.shape)

    np.save(os.path.join(dataset_folder, dataset_name + '_data.npy'), equiv_data)
    np.save(os.path.join(dataset_folder, dataset_name + '_lbls.npy'), equiv_lbls)
    np.save(os.path.join(dataset_folder, dataset_name + '_stabilizers.npy'), equiv_stabilizers)
    np.save(os.path.join(dataset_folder, dataset_name + '_orbit.npy'), orbit_info)
    np.save(os.path.join(dataset_folder, dataset_name + '_class.npy'), class_info)


def generate_eval_data(render_folder, dataset_folder, dataset_name: str, object_type: str, split: str = "train",
                       fixed_number_views: int = 12, resolution: int = 64, seed: int = 28):
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    rng = np.random.default_rng(seed=seed)
    equiv_data = []
    equiv_lbls = []
    equiv_stabilizers = []
    orbit_info = []
    class_info = []

    object_ids = os.listdir(os.path.join(render_folder, split, object_type, "renders"))
    for object_id in object_ids:

        render_path = os.path.join(render_folder, split, object_type, "renders", object_id)
        id_path = os.path.join("/data", render_folder, split, object_type, "identifiers")
        identifiers = os.listdir(id_path)
        with open(os.path.join(id_path, identifiers[0])) as user_file:
            identifiers = json.load(user_file)
        total_views = len(identifiers["filenames"])

        # Select the first view for the object

        random_initial = rng.integers(total_views)

        # Select the available views for the object
        available_views = (random_initial + np.arange(0, total_views,
                                                      total_views // fixed_number_views)) % total_views
        images_per_object = []
        labels_per_object = []
        object_type_per_object = []

        for num_example in range(fixed_number_views):
            index1 = available_views[num_example]
            angle = identifiers["rotations"][index1]
            image1_path = os.path.join(render_path, object_id + "_" + str(index1) + ".png")

            image1 = Image.open(image1_path)
            if resolution != 224:
                image1 = image1.resize((resolution, resolution))
            image1 = process_pil_image(image1)
            images_per_object.append(image1)
            labels_per_object.append(angle)

            print(
                "Generating image with digit number {}, num example {} angle {}".format(object_id, num_example,
                                                                                        angle))
            equiv_stabilizers.append(stabilizer_dict[object_type])
            object_type_per_object.append(OBJECT_DICT_INT[object_type])
        orbit_info.append(OBJECT_DICT_INT[object_type] * ID_MULTIPLIER + int(object_id.split("_")[-1]))
        equiv_data.append(images_per_object)
        equiv_lbls.append(labels_per_object)
        class_info.append(object_type_per_object)

    equiv_data = np.array(equiv_data)
    equiv_lbls = np.array(equiv_lbls)
    equiv_stabilizers = np.array(equiv_stabilizers)
    orbit_info = np.array(orbit_info)
    class_info = np.array(class_info)
    print("Equiv data shape", equiv_data.shape)
    print("Equiv lbls shape", equiv_lbls.shape)
    print("Equiv stabilizers shape", equiv_stabilizers.shape)
    print("Orbit info shape", orbit_info.shape)
    print("Class info shape", class_info.shape)

    np.save(os.path.join(dataset_folder, dataset_name + '_eval_data.npy'), equiv_data)
    np.save(os.path.join(dataset_folder, dataset_name + '_eval_lbls.npy'), equiv_lbls)
    np.save(os.path.join(dataset_folder, dataset_name + '_eval_stabilizers.npy'), equiv_stabilizers)
    np.save(os.path.join(dataset_folder, dataset_name + '_eval_orbit.npy'), orbit_info)
    np.save(os.path.join(dataset_folder, dataset_name + '_eval_class.npy'), class_info)

# if __name__ == "__main__":
