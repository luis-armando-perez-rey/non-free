import os
import numpy as np


def get_square_image(height: int = 64, width: int = 64, n_channels: int = 1, square_size: int = 10) -> np.ndarray:
    """
    Returns an image with a white square placed on the top left corner.
    :param n_channels: number of channels of the square image
    :param height: height of the image
    :param width: width of the image
    :param square_size: size of the square in pixels
    :return:
    """
    pixel_img = np.zeros((height, width, n_channels))
    pixel_img[0:square_size, 0:square_size, :] = 1
    return pixel_img


class ImageTranslation:
    def __init__(self, image: np.ndarray):
        self.image = image
        self.resolution = image.shape[:-1]
        self.channels = image.shape[-1]

    def get_translated_image(self, width_shift, height_shift, pytorch_format=True) -> np.ndarray:
        translated_image = np.roll(self.image, width_shift, axis=-2)
        translated_image = np.roll(translated_image, height_shift, axis=-3)
        if pytorch_format:
            translated_image = np.transpose(translated_image, (2, 0, 1))
        return translated_image


def generate_training_data(images: np.ndarray, n_datapoints: int, dataset_folder: str, dataset_name: str,
                           n_stabilizers: int = 1):
    """
    Generates training data with random translations of image with periodic boundary conditions.
    :param images:
    :param n_datapoints: number of pairs of data to generate
    :param dataset_folder:  folder to save the dataset in
    :param dataset_name:  name of the dataset
    :param n_stabilizers:  number of stabilizers that the input image has default is 1
    :return:
    """
    os.makedirs(dataset_folder, exist_ok=True)
    equiv_data = []
    equiv_actions = []
    equiv_stabilizers = []
    if len(images.shape) == 3:
        print("Only one input image provided")
        images = images[np.newaxis, ...]
    else:
        print("Multiple input images provided, {}".format(images.shape[0]))
    for image in images:
        it = ImageTranslation(image)
        (image_width, image_height, n_channels) = it.image.shape
        for n_datapoint in range(n_datapoints):
            # Get random shifts in width and height for image 1 and 2
            width_shift1 = np.random.randint(0, image_width)
            height_shift1 = np.random.randint(0, image_height)
            width_shift2 = np.random.randint(0, image_width)
            height_shift2 = np.random.randint(0, image_height)
            print("Generating pair {} image1 with shift {} and {}".format(n_datapoint, width_shift1, height_shift1))
            print("Generating pair {} image2 with shift {} and {}".format(n_datapoint, width_shift2, height_shift2))
            # Calculate the translation between image 1 and 2
            width_delta = width_shift2 - width_shift1
            height_delta = height_shift2 - height_shift1
            # Get the images
            image1 = it.get_translated_image(width_shift1, height_shift1)
            image2 = it.get_translated_image(width_shift2, height_shift2)
            # Store the data
            equiv_actions.append([width_delta, height_delta])
            equiv_data.append([image1, image2])
            equiv_stabilizers = ([n_stabilizers, n_stabilizers])

    equiv_data = np.array(equiv_data)
    equiv_lbls = np.array(equiv_actions)

    # Save the dataset
    np.save(os.path.join(dataset_folder, dataset_name + '_data.npy'), equiv_data)
    np.save(os.path.join(dataset_folder, dataset_name + '_lbls.npy'), equiv_lbls)
    np.save(os.path.join(dataset_folder, dataset_name + '_stabilizers.npy'), equiv_stabilizers)
