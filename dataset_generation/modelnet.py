import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.transform import resize
from typing import Tuple, Optional, List
from PIL import Image

stabilizer_dict = {"bathtub": 2, "airplane": 1, "stool": 0, "bench": 3, "bottle":5, "chair":10, "lamp": 1, "bookshelf":2}


def process_pil_image(img):
    img = np.array(img)[:, :, :3]
    img = np.moveaxis(img, -1, 0)
    img = img / 255.0
    return img


render_dictionary = {"bathtub_0": ["bathtub", "0019"],
                     "bathtub_1": ["bathtub", "0020"],
                     "bathtub_2": ["bathtub", "0038"],
                     "bathtub_3": ["bathtub", "0096"],
                     "bathtub_4": ["bathtub", "0108"],
                     # Benches
                     "bench_0": ["bench", "0051"],
                     "bench_1": ["bench", "0116"],
                     "bench_2": ["bench", "0117"],
                     "bench_3": ["bench", "0174"],
                     "bench_4": ["bench", "0175"],
                     # Stools
                     "stool_0": ["stool", "0092"],
                     "stool_0_val": ["stool", "0092"],
                     "stool_1": ["stool", "0100"],
                     "stool_2": ["stool", "0101"],
                     "stool_3": ["stool", "0106"],
                     "airplane_0": ["airplane", "0001"],
                     "airplane_0_val": ["stool", "0092"],
                     "bottle_0": ["bottle", "0001"],
                     "bottle_0_val": ["bottle", "0001"],
                     "chair_0": ["chair", "0889"],
                    "chair_0_val": ["chair", "0889"],
                     "lamp_0": ["lamp", "0122"],
                     "bookshelf_0": ["bookshelf", "0004"],
                     }


def generate_training_data(dataset_folder, dataset_name, object_name: str, object_id: str, split: str = "train",
                           examples_per_object: int = 1, total_angles=360):
    # Dictionary in case we want to convert the digit labels to number of stabilizers

    render_path = os.path.join(".", "data", "modelnet_renders", "renders", object_name, split)
    object_filename_root = object_name + "_" + object_id
    equiv_data = []
    equiv_lbls = []
    equiv_stabilizers = []
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)

    for num_example in range(examples_per_object):
        index1 = np.random.randint(0, total_angles)
        index2 = np.random.randint(0, total_angles)
        angle1 = (360 / total_angles) * index1 * np.pi / 180
        angle2 = (360 / total_angles) * index2 * np.pi / 180
        angle = angle2 - angle1
        image1_path = os.path.join(render_path, object_filename_root + "_" + str(index1) + ".png")
        image2_path = os.path.join(render_path, object_filename_root + "_" + str(index2) + ".png")
        image1 = Image.open(image1_path)
        image2 = Image.open(image2_path)
        image1 = process_pil_image(image1)
        image2 = process_pil_image(image2)

        equiv_data.append([image1, image2])
        equiv_lbls.append(angle)
        print(
            "Generating image with digit number {}, num example {} angle {}".format(object_name, num_example, angle))
        equiv_stabilizers.append(stabilizer_dict[object_name])

    equiv_data = np.array(equiv_data)
    equiv_lbls = np.array(equiv_lbls)
    equiv_stabilizers = np.array(equiv_stabilizers)
    print("Equiv data shape", equiv_data.shape)
    print("Equiv lbls shape", equiv_lbls.shape)
    print("Equiv stabilizers shape", equiv_stabilizers.shape)

    np.save(os.path.join(dataset_folder, dataset_name + '_data.npy'), equiv_data)
    np.save(os.path.join(dataset_folder, dataset_name + '_lbls.npy'), equiv_lbls)
    np.save(os.path.join(dataset_folder, dataset_name + '_stabilizers.npy'), equiv_stabilizers)


def generate_eval_data(dataset_folder, dataset_name, object_name: str, object_id: str, split: str = "train",
                       total_rotations: int = 360):
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    # Regular dataset
    images = []
    stabilizers = []
    labels = []
    angles = np.arange(0, total_rotations, total_rotations // 360)
    render_path = os.path.join(".", "data", "modelnet_renders", "renders", object_name, split)
    object_filename_root = object_name + "_" + object_id

    images_per_object = []
    labels_per_object = []
    for num_angle, angle in enumerate(angles):
        print(
            "Generating image with digit number {}, num example {}".format(object_name, num_angle))
        image_path = os.path.join(render_path, object_filename_root + "_" + str(angle) + ".png")
        image = Image.open(image_path)
        image = process_pil_image(image)

        images_per_object.append(image)
        labels_per_object.append(angle * np.pi / 180)

    stabilizers.append([stabilizer_dict[object_name]] * total_rotations)
    images.append(images_per_object)
    labels.append(labels_per_object)

    images = np.array(images)
    stabilizers = np.array(stabilizers)
    labels = np.array(labels)
    print("Images shape", images.shape)
    print("Stabilizers shape", stabilizers.shape)
    print("Labels shape", labels.shape)
    np.save(os.path.join(dataset_folder, dataset_name + '_eval_data.npy'), images)
    np.save(os.path.join(dataset_folder, dataset_name + '_eval_lbls.npy'), labels)
    np.save(os.path.join(dataset_folder, dataset_name + '_eval_stabilizers.npy'), stabilizers)

# if __name__ == "__main__":
