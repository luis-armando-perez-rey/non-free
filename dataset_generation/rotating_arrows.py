import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.transform import resize
from typing import Tuple, Optional, List

AVAILABLE_TAB_COLORS = ["tab:red", "tab:green", "tab:purple","tab:orange", "tab:blue",   "tab:brown", "tab:pink",
                        "tab:gray", "tab:olive", "tab:cyan"]


class ArrowCanvas:
    def __init__(self, num_arrows: int, style: str, resolution: Tuple[int] = (64, 64), radius: float = 1.0):

        self.num_arrows = num_arrows
        self.style = style
        self.radius = radius
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

    def _add_arrow(self, arrow_angle, color):
        arrow = mpatches.FancyArrowPatch((0, 0), (self.radius * np.cos(arrow_angle), self.radius * np.sin(arrow_angle)),
                                         edgecolor=color, facecolor=color,
                                         mutation_scale=self._mutation_scale,
                                         shrinkA=0,
                                         **self.arrow_params)
        self.ax.add_patch(arrow)

    def add_arrows(self, rotation_rad, color=AVAILABLE_TAB_COLORS[0]):
        for arrow_angle in self._arrow_angles:
            self._add_arrow(arrow_angle + rotation_rad, color)

    def add_multicolor_arrows(self, rotation_rad, colors=None):
        if colors is None:
            colors = AVAILABLE_TAB_COLORS
        assert len(colors) >= len(
            self._arrow_angles), f"Total colors {len(colors)} provided is not enough for {len(self._arrow_angles)} " \
                                 f"arrows "
        for num_arrow, arrow_angle in enumerate(self._arrow_angles):
            self._add_arrow(arrow_angle + rotation_rad, colors[num_arrow])

    @staticmethod
    def show():
        plt.show()


def generate_training_data(num_arrows_list, dataset_folder, dataset_name, style_list: Optional[List[str]] = None,
                           color_list: Optional[List[str]] = None, radius_list: Optional[List[float]] = None,
                           examples_per_num_arrows: int = 100,
                           resolution=(64, 64), multicolor=False):
    equiv_data = []
    equiv_lbls = []
    equiv_stabilizers = []
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    if style_list is None:
        style_list = ["simple", "fancy", "wedge", "fork", "triangle"]
    if color_list is None:
        color_list = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    if radius_list is None:
        radius_list = [1.0]
    for num_arrows in num_arrows_list:
        for style in style_list:
            for color in color_list:
                for radius in radius_list:
                    for num_example in range(examples_per_num_arrows):
                        print(
                            "Generating image with num arrows {}, style {}, color {}, radius {}, multicolor {}, num example {}".format(
                                num_arrows, style, color, radius, multicolor, num_example))
                        c1 = ArrowCanvas(num_arrows=num_arrows, style=style, radius=radius,
                                         resolution=resolution)
                        c2 = ArrowCanvas(num_arrows=num_arrows, style=style, radius=radius,
                                         resolution=resolution)
                        angle1 = 2 * np.pi * np.random.random()
                        angle2 = 2 * np.pi * np.random.random()

                        if multicolor:
                            assert len(
                                color_list) == 1, "When plotting multicolor arrows don't provide a color_list of " \
                                                  "length higher than 1 to avoid unnecesary data creation"
                            c1.add_multicolor_arrows(rotation_rad=angle1)
                            c2.add_multicolor_arrows(rotation_rad=angle2)
                        else:
                            c1.add_arrows(rotation_rad=angle1, color=color)
                            c2.add_arrows(rotation_rad=angle2, color=color)

                        img1 = c1.numpy_torch_img
                        img2 = c2.numpy_torch_img
                        angle = angle2 - angle1

                        equiv_data.append([img1, img2])
                        equiv_lbls.append(angle)
                        equiv_stabilizers.append(num_arrows)
                        plt.close("all")
    equiv_data = np.array(equiv_data)
    equiv_lbls = np.array(equiv_lbls)
    equiv_stabilizers = np.array(equiv_stabilizers)

    np.save(os.path.join(dataset_folder, dataset_name + '_data.npy'), equiv_data)
    np.save(os.path.join(dataset_folder, dataset_name + '_lbls.npy'), equiv_lbls)
    np.save(os.path.join(dataset_folder, dataset_name + '_stabilizers.npy'), equiv_stabilizers)


def generate_training_data_png(num_arrows_list, dataset_folder, dataset_name, style_list: Optional[List[str]] = None,
                               color_list: Optional[List[str]] = None, radius_list: Optional[List[float]] = None,
                               num_angles: int = 36,
                               resolution=(64, 64), multicolor=False):
    equiv_data = []
    equiv_lbls = []
    equiv_stabilizers = []
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    if style_list is None:
        style_list = ["simple", "fancy", "wedge", "fork", "triangle"]
    if color_list is None:
        color_list = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    if radius_list is None:
        radius_list = [1.0]
    for num_arrows in num_arrows_list:
        for style in style_list:
            for color in color_list:
                for radius in radius_list:
                    for num_example in range(examples_per_num_arrows):
                        print(
                            "Generating image with num arrows {}, style {}, color {}, radius {}, multicolor {}, num example {}".format(
                                num_arrows, style, color, radius, multicolor, num_example))
                        c1 = ArrowCanvas(num_arrows=num_arrows, style=style, radius=radius,
                                         resolution=resolution)
                        c2 = ArrowCanvas(num_arrows=num_arrows, style=style, radius=radius,
                                         resolution=resolution)
                        angle1 = np.pi * np.random.random()
                        angle2 = np.pi * np.random.random()

                        if multicolor:
                            assert len(
                                color_list) == 1, "When plotting multicolor arrows don't provide a color_list of " \
                                                  "length higher than 1 to avoid unnecesary data creation"
                            c1.add_multicolor_arrows(rotation_rad=angle1)
                            c2.add_multicolor_arrows(rotation_rad=angle2)
                        else:
                            c1.add_arrows(rotation_rad=angle1, color=color)
                            c2.add_arrows(rotation_rad=angle2, color=color)

                        img1 = c1.numpy_torch_img
                        img2 = c2.numpy_torch_img
                        angle = angle2 - angle1

                        equiv_data.append([img1, img2])
                        equiv_lbls.append(angle)
                        equiv_stabilizers.append(num_arrows)
                        plt.close("all")
    equiv_data = np.array(equiv_data)
    equiv_lbls = np.array(equiv_lbls)
    equiv_stabilizers = np.array(equiv_stabilizers)

    np.save(os.path.join(dataset_folder, dataset_name + '_data.npy'), equiv_data)
    np.save(os.path.join(dataset_folder, dataset_name + '_lbls.npy'), equiv_lbls)
    np.save(os.path.join(dataset_folder, dataset_name + '_stabilizers.npy'), equiv_stabilizers)


def generate_eval_data(num_arrows_list, dataset_folder, dataset_name, style_list: Optional[List[str]] = None,
                       color_list: Optional[List[str]] = None, radius_list: Optional[List[float]] = None,
                       total_rotations: int = 36,
                       resolution=(64, 64), multicolor=False):
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    if style_list is None:
        style_list = ["simple", "fancy", "wedge", "fork", "triangle"]
    if color_list is None:
        color_list = ["red", "green", "blue", "yellow", "cyan", "magenta"]
    if radius_list is None:
        radius_list = [1.0]

    # Regular dataset
    images = []
    stabilizers = []
    for num_arrows in num_arrows_list:
        for style in style_list:
            for color in color_list:
                for radius in radius_list:
                    images_per_object = []
                    for num_angle, angle in enumerate(2 * np.pi * np.linspace(0, 1, total_rotations)):
                        print(num_arrows, num_angle)
                        c = ArrowCanvas(num_arrows=num_arrows, style=style, radius=radius,
                                        resolution=resolution)
                        if multicolor:
                            assert len(
                                color_list) == 1, "When plotting multicolor arrows don't provide a color_list of " \
                                                  "length higher than 1 to avoid unnecesary data creation"
                            c.add_multicolor_arrows(rotation_rad=angle)
                        else:
                            c.add_arrows(rotation_rad=angle, color=color)
                        img = c.numpy_torch_img
                        images_per_object.append(img)
                        plt.close("all")
                    images.append(images_per_object)
                    stabilizers.append([num_arrows] * total_rotations)
    images = np.array(images)
    stabilizers = np.array(stabilizers)
    np.save(os.path.join(dataset_folder, dataset_name + '_eval_data.npy'), images)
    np.save(os.path.join(dataset_folder, dataset_name + '_eval_stabilizers.npy'), stabilizers)
