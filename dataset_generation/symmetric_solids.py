import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import os
from typing import Tuple


def get_reshape_function(resolution: Tuple[int, int] = (64, 64)):
    def reshape_image(example):
        example["image"] = tf.image.resize(example["image"], resolution)
        return example

    return reshape_image


def get_data_from_label(shape_id: int, resolution: Tuple[int, int] = (64, 64), tfds_data_dir="./data", split="train"):
    # Load the data using tensorflow datasets
    ds, ds_info = tfds.load("symmetric_solids", split=split, data_dir=tfds_data_dir, with_info=True)
    reshape_function = get_reshape_function(resolution)
    ds2 = ds.map(reshape_function)
    images = []
    rotations = []
    rotations_equivalent = []
    for example in ds2:
        if example["label_shape"] == shape_id:
            images.append(example["image"])
            rotations.append(example["rotation"])
            rotations_equivalent.append(example["rotations_equivalent"])
    images = np.array(images)
    rotations = np.array(rotations)
    rotations_equivalent = np.array(rotations_equivalent)
    shape_labels = np.ones(len(images)) * shape_id
    return images, rotations, rotations_equivalent, shape_labels


def save_tfds_data_as_npy(save_folder, shape_id, resolution: Tuple[int, int] = (64, 64)):
    os.makedirs(save_folder, exist_ok=True)
    dataset_name = str(int(shape_id))
    if os.path.isfile(os.path.join(save_folder, dataset_name + '_data.npy')):
        print("Data already saved in folder {} for shape id {}".format(save_folder, shape_id))
        return
    print("Loading data from shape {}".format(shape_id))
    images, rotations, rotations_equivalent, shape_labels = get_data_from_label(shape_id, resolution)

    np.save(os.path.join(save_folder, dataset_name + '_data.npy'), images)
    # Save rotations
    np.save(os.path.join(save_folder, dataset_name + '_lbls.npy'), rotations)
    # Save rotations equivalent
    np.save(os.path.join(save_folder, dataset_name + '_rotation_stabilizers.npy'), rotations_equivalent)
    # Save number of stabilizers
    np.save(os.path.join(save_folder, dataset_name + '_stabilizers.npy'),
            np.ones(len(images)) * rotations_equivalent.shape[1])


def generate_training_data(load_folder, shape_id: int, num_pairs: int, resolution: Tuple[int, int] = (64, 64)):
    save_tfds_data_as_npy(load_folder, shape_id, resolution)
    dataset_name = str(int(shape_id))
    loaded_images = np.load(os.path.join(load_folder, dataset_name + "_data.npy"))
    loaded_rotations = np.load(os.path.join(load_folder, dataset_name + "_lbls.npy"))
    images = []
    rotations = []
    for num_pair in range(num_pairs):
        print("Generating pair {}".format(num_pair))
        index1 = np.random.randint(0, len(loaded_images))
        index2 = np.random.randint(0, len(loaded_images))
        image1 = loaded_images[index1]
        image2 = loaded_images[index2]
        rotation1 = loaded_rotations[index1]
        rotation2 = loaded_rotations[index2]
        rotation_delta = rotation2 * np.linalg.inv(rotation1)
        images.append([image1, image2])
        rotations.append(rotation_delta)
    images = np.array(images)
    rotations = np.array(rotations)
    return images, rotations
