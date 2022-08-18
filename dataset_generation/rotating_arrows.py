import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.transform import resize
from typing import Tuple


class ArrowCanvas:
    def __init__(self, num_arrows: int, color: str, style: str, resolution: Tuple[int] = (64, 64), radius: float = 1.0):

        self.num_arrows = num_arrows
        self.style = style
        self.radius = radius
        self.color = color
        self.resolution = resolution
        self.fig, self.ax = self._set_figax()

    @staticmethod
    def _set_figax():
        fig, ax = plt.subplots(1, figsize=(5, 5))
        ax.margins(0)
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
        fig.tight_layout()
        return fig, ax

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, radius):
        if self.style == "fork":
            self._radius = radius * 0.8
        else:
            self._radius = radius

    @property
    def numpy_img(self):
        self.fig.canvas.draw()
        array = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        array = array.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
        return resize(array, self.resolution)

    @property
    def numpy_torch_img(self):
        """
        Returns a numpy with the torch format (channels, height, width)
        :return:
        """
        return np.transpose(self.numpy_img, (2, 0, 1))

    @property
    def _mutation_scale(self):
        if self.num_arrows == 1:
            return 200
        else:
            return 100

    @property
    def _arrow_angles(self):
        angles = 2 * np.pi * np.linspace(0, 1, self.num_arrows, endpoint=False)
        return angles

    @property
    def arrow_params(self):
        if self.style == "simple":
            params = dict(linewidth=0.0,
                          arrowstyle=mpatches.ArrowStyle("simple"))
        elif self.style == "fancy":
            params = dict(linewidth=0.0,
                          arrowstyle=mpatches.ArrowStyle("fancy", head_length=0.5, head_width=0.5, tail_width=0.3))
        elif self.style == "wedge":
            params = dict(linewidth=0.0,
                          arrowstyle=mpatches.ArrowStyle("wedge", head_length=0.5, tail_width=0.3))
        elif self.style == "fork":
            params = dict(linewidth=10.0,
                          arrowstyle=mpatches.ArrowStyle("-[", widthB=0.2, lengthB=0.2))
        elif self.style == "triangle":
            params = dict(linewidth=0.0,
                          arrowstyle=mpatches.ArrowStyle("-|>", head_length=0.5))
        else:
            raise ValueError("Invalid arrow style, valid options are 'simple', 'fancy', 'wedge', 'fork', 'triangle'")
        return params

    def _add_arrow(self, arrow_angle):
        arrow = mpatches.FancyArrowPatch((0, 0), (self.radius * np.cos(arrow_angle), self.radius * np.sin(arrow_angle)),
                                         edgecolor=self.color, facecolor=self.color,
                                         mutation_scale=self._mutation_scale,
                                         shrinkA=0,
                                         **self.arrow_params)
        self.ax.add_patch(arrow)

    def add_arrows(self, rotation_rad):
        for arrow_angle in self._arrow_angles:
            self._add_arrow(arrow_angle + rotation_rad)

    @staticmethod
    def show():
        plt.show()


N = 20000

equiv_data = []
equiv_lbls = []

dataset_folder = "../data/arrows"
os.makedirs(dataset_folder, exist_ok=True)

for i in range(N):
    print(i)
    c1 = ArrowCanvas(num_arrows=4, color="tab:red", style="simple")
    c2 = ArrowCanvas(num_arrows=4, color="tab:red", style="simple")
    angle1 = np.pi * np.random.random()
    angle2 = np.pi * np.random.random()

    c1.add_arrows(rotation_rad=angle1)
    c2.add_arrows(rotation_rad=angle2)

    img1 = c1.numpy_torch_img
    img2 = c2.numpy_torch_img
    angle = angle2 - angle1

    equiv_data.append([img1, img2])
    equiv_lbls.append(angle)
    plt.close("all")

equiv_data = np.array(equiv_data)
equiv_lbls = np.array(equiv_lbls)
print(equiv_data.shape, equiv_lbls.shape)

np.save(os.path.join(dataset_folder, 'equiv_data.npy'), equiv_data)
np.save(os.path.join(dataset_folder, 'equiv_lbls.npy'), equiv_lbls)
