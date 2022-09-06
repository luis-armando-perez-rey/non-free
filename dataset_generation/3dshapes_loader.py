# Code from the dsprites github repo www.github.com/deepmind/3d-shapes
# Please cite this paper if you use this code.
# @misc{3dshapes18,
#   title={3D Shapes Dataset},
#   author={Burgess, Chris and Kim, Hyunjik},
#   howpublished={https://github.com/deepmind/3dshapes-dataset/},
#   year={2018}
# }
import numpy as np
import matplotlib.pyplot as plt
import h5py

FILEPATH = "../data/3dshapes/3dshapes.h5"
_FACTORS_IN_ORDER = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape',
                     'orientation']
_NUM_VALUES_PER_FACTOR = {'floor_hue': 10, 'wall_hue': 10, 'object_hue': 10,
                          'scale': 8, 'shape': 4, 'orientation': 15}


def load_data(filename):
    print("Loading 3dshapes dataset...")
    dataset = h5py.File(filename, "r")
    images = dataset["images"]
    factors = dataset["labels"]
    return images, factors


imgs, factors = load_data(FILEPATH)


# Define number of values per latents and functions to convert to indices


def get_index(factors):
    """ Converts factors to indices in range(num_data)
    Args:
      factors: np array shape [6,batch_size].
               factors[i]=factors[i,:] takes integer values in
               range(_NUM_VALUES_PER_FACTOR[_FACTORS_IN_ORDER[i]]).

    Returns:
      indices: np array shape [batch_size].
    """
    indices = 0
    base = 1
    for factor, name in reversed(list(enumerate(_FACTORS_IN_ORDER))):
        print(factor, name, indices, base)
        indices += factors[factor] * base
        base *= _NUM_VALUES_PER_FACTOR[name]
    return indices


def get_images_shapes(shape_ids=None):
    print("Getting 3dshapes images per shape")
    images, factors = load_data(FILEPATH)
    if shape_ids is None:
        shape_ids = [0, 1, 2, 3]
    output_images = []
    print(images.shape)
    print(factors.shape)
    for id in shape_ids:
        selection_factor = np.expand_dims([0, 0, 0, 1, id, 0], axis=-1)
        print(selection_factor.shape)
        index = get_index(selection_factor)
        print("Index", get_index(selection_factor))
        print(index.shape)
        output_images.append(images[get_index(selection_factor)])
        output_images = np.array(np.expand_dims(output_images, axis=-1))
    return output_images


if __name__ == "__main__":
    shape_images = get_images_shapes(imgs)

    fig, axes = plt.subplots(1, 3, figsize=(10, 5))
    for num_ax, ax in enumerate(axes):
        ax.imshow(shape_images[num_ax, :, :, 0])
        ax.set_title('Image {}'.format(num_ax))
        ax.axis('off')
    plt.tight_layout()
    plt.show()
