import torch
import numpy as np
import os
from typing import Tuple
from models.classifiers import get_entropy_classifier, rebase_labels


def get_efficient_stochastic_prediction(z: torch.Tensor, model: torch.nn.Module, num_repetitions: int = 100):
    z_repeated = torch.tile(z, (num_repetitions, 1))


def load_data(load_dir: str, identifier: str = ""):
    """
    Load data from the load_dir corresponding to the identifier
    :param load_dir: path to the directory
    :param identifier: identifier of the file
    :return:
    """
    data = np.load(os.path.join(load_dir, identifier + ".npy"))
    return data


def load_embeddings(load_dir: str, identifier: str = ""):
    """
    Load embeddings from the load_dir corresponding to the mean in the circle, the logvar, extra embedding
    and the labels
    :param load_dir: path to the directory
    :param identifier: identifier of the files
    :return:
    """
    mean_circle = load_data(load_dir, identifier + "_mean")
    logvar_eval = load_data(load_dir, identifier + "_logvar")
    extra = load_data(load_dir, identifier + "_extra")
    labels = load_data(load_dir, identifier + "_labels")
    return mean_circle, logvar_eval, extra, labels


def expand_labels(labels: np.array, n_means: int) -> torch.Tensor:
    """
    Expand the labels to match the number of means
    :param labels: labels numpy array
    :param n_means: number of means from EquIN
    :return:
    """
    repeat_array_along_axis(labels, n_means, axis=-1)
    labels = labels.reshape((-1))
    return torch.tensor(labels)


def repeat_array_along_axis(data: np.array, repetitions: int, axis=-1) -> np.array:
    """
    Repeat the array along an axis
    :param data: array that needs to be expanded and repeated
    :param repetitions: number of repetitions
    :param axis: axis that needs to be expanded and repeated
    :return:
    """
    data = np.expand_dims(data, axis=axis)
    repetition_array = np.ones(len(data.shape), dtype=int)
    repetition_array[axis] = repetitions
    data = np.tile(data, repetition_array)
    return data


def repeat_tensor_along_axis(data: torch.Tensor, repetitions: int, axis=-1) -> torch.Tensor:
    """
    Repeat the array along an axis
    :param data: array that needs to be expanded and repeated
    :param repetitions: number of repetitions
    :param axis: axis that needs to be expanded and repeated
    :return:
    """
    data = data.unsqueeze(axis)
    repetition_tuple = np.ones(len(data.shape), dtype=int)
    repetition_tuple[axis] = repetitions
    repetition_tuple = tuple(repetition_tuple)
    data = torch.tile(data, repetition_tuple)
    return data


def create_extra_fixed(extra, total_views, n_means, num_fixed_extra):
    extra_fixed = np.copy(extra)
    extra_fixed = extra_fixed.reshape((-1, total_views, extra_fixed.shape[-1]))[:, num_fixed_extra, :]
    extra_fixed = repeat_array_along_axis(extra_fixed, total_views, axis=1)
    extra_fixed = extra_fixed.reshape((-1, extra_fixed.shape[-1]))
    extra_fixed = repeat_array_along_axis(extra_fixed, n_means, axis=1)
    return extra_fixed


def get_latent_representations(mean_circle, extra):
    """
    Get the latent representations from the mean in the circle and the extra embedding joined together
    :param mean_circle: mean from the circle obtained as output from EquIN
    :param extra: extra embedding obtained as output from EquIN
    :return:
    """
    n_means = mean_circle.shape[1]
    extra_expanded = repeat_tensor_along_axis(extra, n_means, axis=1)
    latent_representation = np.concatenate([mean_circle, extra_expanded], axis=-1)
    return latent_representation


# region EVALUATION UTILS
def get_accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Get the accuracy of the model
    :param outputs: outputs as probabilities from softmax
    :param labels: rebased labels (starting from 0)
    :return:
    """
    correct = get_true_positives(outputs, labels)
    return correct / labels.size(0)


def get_true_positives(outputs, labels):
    """
    Get the true positives of the model
    :param outputs: outputs as probabilities from softmax
    :param labels: rebased labels (starting from 0)
    :return:
    """
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).int()
    return correct


def order_views_based_indices(numpy_array: np.array, indices: np.array):
    """
    Order the views based on the indices
    :param numpy_array: numpy array with shape (dim1,dim2,...,num_views,...
    :param indices: indices with shape (dim1, dim2)
    :return:
    """
    dim1, dim2 = indices.shape
    index_range = np.arange(dim1)
    ordered_outputs = numpy_array[index_range[:, np.newaxis], indices, ...]
    return ordered_outputs


def order_views_based_indices_tensor(tensor: torch.Tensor, indices: torch.Tensor):
    """
    Order the views based on the indices
    :param tensor: torch tensor with shape (dim1, dim2, ..., num_views, ...)
    :param indices: indices with shape (dim1, dim2)
    :return:
    """
    dim1, dim2 = indices.shape
    index_range = torch.arange(dim1)
    ordered_outputs = tensor[index_range[:, None], indices, ...]
    del(index_range)
    return ordered_outputs

# def order_views_based_indices_tensor(tensor: torch.Tensor, indices: np.array):
#     """
#     Order the views based on the indices
#     :param tensor:
#     :param indices:
#     :return:
#     """
#     ordered_outputs = []
#     for num_ordered_index, ordered_index in enumerate(indices):
#         ordered_outputs.append(tensor[num_ordered_index, ordered_index, ...])
#     return torch.stack(ordered_outputs, dim=0)


# endregion

# region PREDICTIONS
def get_list_of_random_indices(data_array):
    # Data_array has shape (dim1, dim2) create an array of dimensions (dim1, dim2) with random non-repeating indices for dim2
    # Example: data_array = np.array([[1,2,3],[4,5,6]])
    # Output: np.array([[0,2,1],[2,1,0]])

    # Get the dimensions of the data array
    dim1, dim2 = data_array.shape

    # Create an array of dimensions (dim1, dim2) with random non-repeating indices for dim2
    random_indices = np.apply_along_axis(np.random.permutation, 1, np.tile(np.arange(dim2), (dim1, 1)))

    return random_indices

def get_list_of_random_indices_tensor(data_tensor):
    """
    Create a tensor of dimensions (dim1, dim2) with random non-repeating indices for dim2
    :param data_tensor: torch tensor with shape (dim1, dim2)
    :return: torch tensor with shape (dim1, dim2) containing random non-repeating indices for dim2
    """
    # Get the dimensions of the data tensor
    dim1, dim2 = data_tensor.shape

    # Create a tensor of dimensions (dim1, dim2) with random non-repeating indices for dim2
    random_indices = torch.stack([torch.randperm(dim2) for _ in range(dim1)])

    return random_indices


def get_stochastic_prediction(z: torch.Tensor, model: torch.nn.Module, num_repetitions: int = 100) -> torch.Tensor:
    """
    Get the stochastic prediction of the model by predicting repetitions of the input. This method
    is useful for models that are stochastic
    :param z: input
    :param model: model to use for predictions predict.
    :param num_repetitions: number of repetitions
    :return:
    """
    z = z.unsqueeze(1)
    repetitions = np.ones(len(z.shape), dtype=int)
    repetitions[1] = num_repetitions
    repetitions = tuple(repetitions)
    z = z.repeat(repetitions)
    prediction = torch.softmax(model(z), dim=-1)
    prediction = prediction.mean(1)
    del(z)
    return prediction


def get_expected_entropy_dropout(z: torch.Tensor, model: torch.nn.Module, num_repetitions: int = 100) -> torch.Tensor:
    """
    Get the expected entropy of the model by predicting repetitions of the input.
    :param z: input
    :param model: model to use for predictions predict.
    :param num_repetitions: number of repetitions
    :return:
    """
    z = repeat_tensor_along_axis(z, num_repetitions, axis=1)
    entropies = get_entropy_classifier(z, model)
    entropies = entropies.mean(1)
    return entropies


def predict_probability(z: torch.Tensor, clsf: torch.nn.Module):
    """
    Predict the probability of the model that outputs logits
    :param z: input to the model
    :param clsf: classifier model
    :return:
    """
    return torch.softmax(clsf(z), dim=-1)


def calculate_accuracy_objects(reshaped_circular: torch.Tensor, reshaped_extra: torch.Tensor,
                               reshaped_labels: torch.Tensor, view_indices: np.array, model: torch.tensor,
                               num_repetitions: int = 100) -> torch.Tensor:
    """
    Calculate the accuracy of the model for each object
    :param reshaped_circular: tensor of shape (num_objects, n_means, 2)
    :param reshaped_extra: tensor of shape (num_objects, n_means, extra_dim)
    :param reshaped_labels: tensor of shape (num_objects, n_means, 1)
    :param view_indices: indices of the views that indicate their order
    :param model: model to use for predictions
    :param num_repetitions: number of repetitions to use for stochastic models
    :return:
    """
    # Estimate the number of objects and the number of means
    num_objects = reshaped_circular.shape[0]
    n_means = reshaped_circular.shape[1]

    # Order the input data based on the indices
    circular_view = order_views_based_indices_tensor(reshaped_circular, view_indices)
    extra_view = order_views_based_indices_tensor(reshaped_extra, view_indices)
    label_view = order_views_based_indices_tensor(reshaped_labels, view_indices)[..., 0, 0]

    # Join the circular and extra embeddings
    latent_view = torch.concatenate([circular_view, extra_view], axis=-1)
    output = get_stochastic_prediction(torch.tensor(latent_view).float(), model, num_repetitions)
    output = output.mean(1)
    accuracy = get_accuracy(output, rebase_labels(label_view))
    return accuracy

# endregion
