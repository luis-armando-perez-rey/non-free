import numpy as np


def torch_data_to_numpy(data):
    """
    Convert torch data to numpy data
    :param data:
    :return:
    """
    # Assert if dat is already a numpy array if so, do nothing.
    if isinstance(data, np.ndarray):
        return data
    # When data has a single dimension (sinusoidal 1D data)
    if data.dim() == 2:
        return data.detach().cpu().numpy()
    # When data are images
    else:
        return np.squeeze(np.transpose(data.detach().cpu().numpy(), axes=[0, 2, 3, 1]))
