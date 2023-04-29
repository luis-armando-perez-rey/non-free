import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional


class SimpleClassifier(nn.Module):
    def __init__(self, input_dim_size: int, hidden_layers_list: List[int], num_classes: int):
        """
        Simple classifier with dense layers and ReLU activations
        :param input_dim_size: input data dimensions
        :param hidden_layers_list: list of layersizes
        :param num_classes: output number of classes
        """
        super().__init__()
        layer_list = [nn.Linear(input_dim_size, hidden_layers_list[0])]
        for i in range(len(hidden_layers_list) - 1):
            layer_list.append(nn.Linear(hidden_layers_list[i], hidden_layers_list[i + 1]))
        self.layer_list = nn.ModuleList(layer_list)

        self.final_layer = nn.Linear(hidden_layers_list[-1], num_classes)

    def forward(self, x):
        for layer in self.layer_list:
            x = nn.functional.relu(layer(x))
        x = self.final_layer(x)
        return x


class SimpleClassifierDropout(nn.Module):
    def __init__(self, input_dim_size: int, hidden_layers_list: List[int], num_classes: int,
                 dropout_p_list: Optional[List[float]] = None):
        """
        Simple classifier with dense layers and ReLU activations
        :param input_dim_size: input data dimensions
        :param hidden_layers_list: list of layersizes
        :param num_classes: output number of classes
        """
        if dropout_p_list is None:
            dropout_p_list = np.ones_like(hidden_layers_list) * 0.2
        assert len(dropout_p_list) == len(hidden_layers_list)
        super().__init__()
        layer_list = [nn.Linear(input_dim_size, hidden_layers_list[0])]
        dropout_list = [nn.Dropout(dropout_p_list[0])]
        for i in range(len(hidden_layers_list) - 1):
            layer_list.append(nn.Linear(hidden_layers_list[i], hidden_layers_list[i + 1]))
            dropout_list.append(nn.Dropout(dropout_p_list[i + 1]))
        self.layer_list = nn.ModuleList(layer_list)
        self.dropout_list = nn.ModuleList(dropout_list)
        self.final_layer = nn.Linear(hidden_layers_list[-1], num_classes)

    def forward(self, x):
        for num_layer, layer in enumerate(self.layer_list):
            x = nn.functional.relu(self.dropout_list[num_layer](layer(x)))
        x = self.final_layer(x)
        return x


def get_categorical(z, clf):
    """
    Takes z as logits and produces a distribution from the output of the clf classifier
    """
    probs = torch.softmax(clf(z), dim=-1)
    return torch.distributions.Categorical(probs)


def get_entropy_gradients(z, clf):
    """
    Calculates the gradients of the entropy of the categorical distribution with respect to the input
    :param z: 
    :param clf: 
    :return: 
    """
    z_values_copy = z.clone().detach().requires_grad_(True)
    p_dist = get_categorical(z_values_copy, clf)
    entropy = p_dist.entropy()
    # Get the gradient of the entropy with respect to each input
    entropy.backward(torch.ones_like(entropy), retain_graph=True)
    # Get the gradients
    gradients = z_values_copy.grad.clone().detach().requires_grad_(False)
    return gradients


def one_hot_encode(labels: torch.Tensor) -> torch.Tensor:
    """
    One hot encodes the labels
    :param labels: tensor with integer labels
    :return:
    """
    # Get the number of classes
    num_classes = len(torch.unique(labels))
    # Rebase the labels to start from 0
    rebased_labels = rebase_labels(labels).unsqueeze(1)
    # Create the one hot encoding
    one_hot = torch.zeros(labels.size(0), num_classes)
    # Fill the one hot encoding
    one_hot.scatter_(1, rebased_labels, 1)
    return one_hot


def rebase_labels(labels: torch.Tensor) -> torch.Tensor:
    """
    Rebase the labels to start from 0
    :param labels: tensor with integer labels
    :return:
    """
    # Find the unique labels
    unique_labels = torch.unique(labels)
    # Create a dictionary to map the labels to the indices
    label_to_index = {label.item(): index for index, label in enumerate(unique_labels)}
    # Rebase the labels to start from 0
    rebased_labels = torch.tensor([label_to_index[label.item()] for label in labels])
    return rebased_labels
