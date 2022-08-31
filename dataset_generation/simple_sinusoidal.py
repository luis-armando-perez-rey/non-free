import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.transform import resize
from typing import Tuple, Optional, List

AVAILABLE_TAB_COLORS = ["tab:red", "tab:green", "tab:purple", "tab:orange", "tab:blue", "tab:brown", "tab:pink",
                        "tab:gray", "tab:olive", "tab:cyan"]


class SinusoidalData:
    def __init__(self, dimension: int):
        self.num_points = dimension

    @property
    def domain_points(self):
        return np.linspace(0, 1, self.num_points) * 2 * np.pi

    def generate_sinusoidal(self, omega: int, phase: float, noise_std: float = 0.0,
                            amplitude: float = 1.0) -> np.ndarray:
        # assert if omega is an integer
        assert isinstance(omega, int)
        # assert if phase is between 0 and np.pi
        assert 0 <= phase <= 2 * np.pi, "Phase must be between 0 and 2pi"
        # assert if noise_std is a positive number
        assert noise_std >= 0, "Noise std must be a positive number"
        # generate sinusoidal signal
        sinusoidal = amplitude * np.sin(omega * (self.domain_points + phase))
        # add noise if noise_std is not 0
        if noise_std != 0:
            sinusoidal += np.random.normal(0, noise_std, sinusoidal.shape)
        return sinusoidal

def plot_sinusoidal(sinusoidal):
    domain_points = np.linspace(0, 1, sinusoidal.shape[-1]) * 2 * np.pi
    plt.plot(domain_points, sinusoidal)
    plt.show()


def generate_dataset_sinusoidals(omega_list: List[int], dimension: int, dataset_folder: str,
                                 dataset_name: str, num_examples: int = 10000):
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    equiv_data = []
    equiv_lbls = []
    equiv_stabilizers = []
    sinusoidal = SinusoidalData(dimension)
    for omega in omega_list:
        for num_example in range(num_examples):
            print("Generating omega {} example {}".format(omega, num_example))
            angle1 = 2 * np.pi * np.random.random()
            angle2 = 2 * np.pi * np.random.random()
            s1 = sinusoidal.generate_sinusoidal(omega, angle1)
            s2 = sinusoidal.generate_sinusoidal(omega, angle2)
            angle = angle2 - angle1

            equiv_data.append([s1, s2])
            equiv_lbls.append(angle)
            equiv_stabilizers.append(omega)

    equiv_data = np.array(equiv_data)
    equiv_lbls = np.array(equiv_lbls)
    equiv_stabilizers = np.array(equiv_stabilizers)

    np.save(os.path.join(dataset_folder, dataset_name + '_data.npy'), equiv_data)
    np.save(os.path.join(dataset_folder, dataset_name + '_lbls.npy'), equiv_lbls)
    np.save(os.path.join(dataset_folder, dataset_name + '_stabilizers.npy'), equiv_stabilizers)


def generate_dataset_regular_sinusoidals(omega_list: List[int], dimension: int, num_angles:int, dataset_folder: str,
                                         dataset_name: str):
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    equiv_data = []
    equiv_stabilizers = []
    sinusoidal = SinusoidalData(dimension)
    for omega in omega_list:
        datapoints_per_omega = []
        stabilizers_per_omega = []
        for angle in np.linspace(0, 2 * np.pi, num_angles, endpoint=False):
            s = sinusoidal.generate_sinusoidal(omega, angle)
            datapoints_per_omega.append(s)
            stabilizers_per_omega.append(omega)
        equiv_data.append(datapoints_per_omega)
        equiv_stabilizers.append(stabilizers_per_omega)

    equiv_data = np.array(equiv_data)
    equiv_stabilizers = np.array(equiv_stabilizers)

    np.save(os.path.join(dataset_folder, dataset_name + '_eval_data.npy'), equiv_data)
    np.save(os.path.join(dataset_folder, dataset_name + '_eval_stabilizers.npy'), equiv_stabilizers)

