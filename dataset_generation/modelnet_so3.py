import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.transform import resize
from typing import Tuple, Optional, List
from scipy.spatial.transform import Rotation as R
from PIL import Image

stabilizer_dict = {"bathtub": 2, "airplane": 1, "stool": 0, "bench": 3, "bottle": 5, "chair": 10, "lamp": 1,
                   "bookshelf": 2}


def process_pil_image(img):
    img = np.array(img)[:, :, :3]
    img = np.moveaxis(img, -1, 0)
    img = img / 255.0
    return img

def generate_data(load_folder, save_folder, dataset_name, split="train", object_type="airplane", object_id="airplane_0001"):
    os.makedirs(save_folder, exist_ok=True)
    training_pairs = 2000
    eval_pairs = 250
    dataset_folder = os.path.join(load_folder, "renders_so3",object_type, split)
    identifiers_folder = os.path.join(dataset_folder, "identifiers")
    images_folder = os.path.join(dataset_folder, "images")


    # Load the rotations
    quaternions = R.from_quat(np.load(os.path.join(identifiers_folder, "rot_" + object_id + ".npy")))
    rot_matrices = quaternions.as_matrix()

    # Generate training data
    num_training_pairs = np.arange(training_pairs)
    num_eval_pairs = np.arange(training_pairs, training_pairs + eval_pairs)
    print("Generating training data")
    images, stabilizers, rotations = get_data_pairs(images_folder, rot_matrices, num_training_pairs)
    save_data(images, stabilizers, rotations, save_folder, dataset_name)
    print("Generating eval data")
    images_val, stabilizers_val, rotations_val = get_data_pairs(images_folder, rot_matrices, num_eval_pairs)
    save_data(images_val, stabilizers_val, rotations_val, save_folder, dataset_name+"_val")
    print(images.shape, stabilizers.shape, rotations.shape)

def save_data(images, stabilizers, rotations, save_folder, dataset_name):
    np.save(os.path.join(save_folder, dataset_name + '_data.npy'), images)
    # Save rotations
    np.save(os.path.join(save_folder, dataset_name + '_lbls.npy'), rotations)
    # Save number of stabilizers
    np.save(os.path.join(save_folder, dataset_name + '_stabilizers.npy'), stabilizers)


def get_data_pairs(images_folder, rot_matrices, pair_arange):
    # Generate training data
    images = []
    rotations = []

    for num_training_pair in pair_arange:
        print("Generating image pair {}".format(num_training_pair))
        # Select images as even and odd
        index1 = 2 * num_training_pair
        index2 = 2 * num_training_pair + 1
        image1 = Image.open(os.path.join(images_folder, "airplane_0001_" + str(index1) + ".png"))
        image2 = Image.open(os.path.join(images_folder, "airplane_0001_" + str((index2) + 1) + ".png"))
        image1 = process_pil_image(image1)
        image2 = process_pil_image(image2)

        rotation1 = rot_matrices[index1]
        rotation2 = rot_matrices[index2]
        rotation_delta = rotation2 @ np.linalg.inv(rotation1)
        images.append([image1, image2])
        rotations.append(rotation_delta)
    stabilizers = np.array([1] * len(pair_arange))
    rotations = np.array(rotations)
    images = np.array(images)
    return images, stabilizers, rotations


if __name__ == "__main__":
    generate_data(dataset_name = "test", load_folder="../data/modelnet_renders", save_folder="../data/modelnet_so3")
