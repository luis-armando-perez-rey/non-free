# Code from the dsprites github repo www.github.com/deepmind/dsprites-dataset
# Please cite this paper if you use this code.
# @misc{dsprites17,
# author = {Loic Matthey and Irina Higgins and Demis Hassabis and Alexander Lerchner},
# title = {dSprites: Disentanglement testing Sprites dataset},
# howpublished= {https://github.com/deepmind/dsprites-dataset/},
# year = "2017",
# }
import numpy as np
import matplotlib.pyplot as plt

FILEPATH = "./data/dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"


def load_data(filename):
    print("Loading dsprites dataset...")
    dataset_zip = np.load(filename, allow_pickle=True, encoding='latin1')
    images = dataset_zip['imgs']
    latent_values = dataset_zip["latents_values"]
    latent_values_dict = dict(color=np.unique(latent_values[:, 0]),
                              shape=np.unique(latent_values[:, 1]),
                              scale=np.unique(latent_values[:, 2]),
                              orientation=np.unique(latent_values[:, 3]),
                              posx=np.unique(latent_values[:, 4]),
                              posy=np.unique(latent_values[:, 5]))
    latent_values_dict["latent_sizes"] = np.array([len(latent_values_dict[key]) for key in latent_values_dict.keys()])
    return images, latent_values_dict


imgs, latent_dict = load_data(FILEPATH)
# Define number of values per latents and functions to convert to indices
latents_sizes = latent_dict['latent_sizes']
latents_bases = np.concatenate((latents_sizes[::-1].cumprod()[::-1][1:],
                                np.array([1, ])))


def latent_to_index(latents):
    return np.dot(latents, latents_bases).astype(int)


def get_images_shapes(shape_ids=None):
    print("Getting dsprites images per shape")
    images, latent_values_dict = load_data(FILEPATH)
    if shape_ids is None:
        shape_ids = [0, 1, 2]
    output_images = []
    for id in shape_ids:
        output_images.append(images[latent_to_index([0, id, 1, 0, 0, 0])])
    output_images = np.array(np.expand_dims(output_images, axis=-1))
    return output_images


if __name__ == "__main__":
    shape_images = get_images_shapes(imgs, [0, 1, 2])

    fig, axes = plt.subplots(1, 3, figsize=(10, 5))
    for num_ax, ax in enumerate(axes):
        ax.imshow(shape_images[num_ax, :, :, 0])
        ax.set_title('Image {}'.format(num_ax))
        ax.axis('off')
    plt.tight_layout()
    plt.show()
