import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.transform import resize
from typing import Tuple, Optional, List

AVAILABLE_TAB_COLORS = ["tab:red", "tab:green", "tab:purple", "tab:orange", "tab:blue", "tab:brown", "tab:pink",
                        "tab:gray", "tab:olive", "tab:cyan"]


class ArrowCanvas:
    def __init__(self, resolution: Tuple[int] = (64, 64)):
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

    @staticmethod
    def get_mutation_scale(num_arrows):
        if num_arrows == 1:
            return 100
        else:
            return 100

    @staticmethod
    def get_arrow_angles(num_arrows):
        angles = 2 * np.pi * np.linspace(0, 1, num_arrows, endpoint=False)
        return angles

    @staticmethod
    def get_arrow_params(style):
        if style == "simple":
            params = dict(linewidth=0.0,
                          arrowstyle=mpatches.ArrowStyle("simple"))
        elif style == "fancy":
            params = dict(linewidth=0.0,
                          arrowstyle=mpatches.ArrowStyle("fancy", head_length=0.5, head_width=0.5, tail_width=0.3))
        elif style == "wedge":
            params = dict(linewidth=0.0,
                          arrowstyle=mpatches.ArrowStyle("wedge", tail_width=0.3))
        elif style == "fork":
            params = dict(linewidth=10.0,
                          arrowstyle=mpatches.ArrowStyle("-[", widthB=0.2, lengthB=0.2))
        elif style == "triangle":
            params = dict(linewidth=0.0,
                          arrowstyle=mpatches.ArrowStyle("-|>", head_length=0.5))
        else:
            raise ValueError("Invalid arrow style, valid options are 'simple', 'fancy', 'wedge', 'fork', 'triangle'")
        return params

    def _add_arrow(self, arrow_angle, num_arrows, color, start_point=(0, 0), radius=1.0, style="simple"):
        arrow_params = self.get_arrow_params(style)
        mutation_scale = self.get_mutation_scale(num_arrows)
        arrow = mpatches.FancyArrowPatch(start_point, (
            start_point[0] + radius * np.cos(arrow_angle), start_point[1] + radius * np.sin(arrow_angle)),
                                         edgecolor=color, facecolor=color,
                                         mutation_scale=mutation_scale,
                                         shrinkA=0,
                                         **arrow_params)
        self.ax.add_patch(arrow)

    def add_arrows(self, rotation_rad, num_arrows, color=AVAILABLE_TAB_COLORS[0], start_point=(0, 0), radius=1.0,
                   style="simple"):
        arrow_angles = self.get_arrow_angles(num_arrows)
        for arrow_angle in arrow_angles:
            self._add_arrow(arrow_angle + rotation_rad, num_arrows, color, start_point, radius, style)

    def add_multicolor_arrows(self, rotation_rad, num_arrows, colors=None, start_point=(0, 0), radius=1.0,
                              style="simple"):
        arrow_angles = self.get_arrow_angles(num_arrows)
        if colors is None:
            colors = AVAILABLE_TAB_COLORS
        assert len(colors) >= len(arrow_angles), f"Total colors {len(colors)} provided is not enough for " \
                                                 f"{len(arrow_angles)} arrows "
        for num_arrow, arrow_angle in enumerate(arrow_angles):
            self._add_arrow(arrow_angle + rotation_rad, num_arrows, colors[num_arrow], start_point, radius, style)

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
                            "Generating image with num arrows {}, style {}, color {}, radius {}, multicolor {}, "
                            "num example {}".format(
                                num_arrows, style, color, radius, multicolor, num_example))
                        c1 = ArrowCanvas(resolution=resolution)
                        c2 = ArrowCanvas(resolution=resolution)
                        angle1 = 2 * np.pi * np.random.random()
                        angle2 = 2 * np.pi * np.random.random()

                        if multicolor:
                            assert len(
                                color_list) == 1, "When plotting multicolor arrows don't provide a color_list of " \
                                                  "length higher than 1 to avoid unnecesary data creation"
                            c1.add_multicolor_arrows(rotation_rad=angle1, num_arrows=num_arrows, colors=None,
                                                     radius=radius, style=style)
                            c2.add_multicolor_arrows(rotation_rad=angle2, num_arrows=num_arrows, colors=None,
                                                     radius=radius, style=style)
                        else:
                            c1.add_arrows(rotation_rad=angle1, num_arrows=num_arrows, color=color,
                                          radius=radius, style=style)
                            c2.add_arrows(rotation_rad=angle2, num_arrows=num_arrows, color=color,
                                          radius=radius, style=style)

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
    print("Equiv data shape", equiv_data.shape)
    print("Equiv lbls shape", equiv_lbls.shape)
    print("Equiv stabilizers shape", equiv_stabilizers.shape)

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
    labels = []
    for num_arrows in num_arrows_list:
        for style in style_list:
            for color in color_list:
                for radius in radius_list:
                    images_per_object = []
                    labels_per_object = []
                    for num_angle, angle in enumerate(2 * np.pi * np.linspace(0, 1, total_rotations)):
                        print(num_arrows, num_angle)
                        c = ArrowCanvas(resolution=resolution)
                        if multicolor:
                            assert len(
                                color_list) == 1, "When plotting multicolor arrows don't provide a color_list of " \
                                                  "length higher than 1 to avoid unnecesary data creation"
                            c.add_multicolor_arrows(rotation_rad=angle, num_arrows=num_arrows, colors=None,
                                                    radius=radius, style=style)
                        else:
                            c.add_arrows(rotation_rad=angle, num_arrows=num_arrows, color=color,
                                         radius=radius, style=style)
                        img = c.numpy_torch_img
                        images_per_object.append(img)
                        plt.close("all")
                        labels_per_object.append(angle)
                    images.append(images_per_object)
                    stabilizers.append([num_arrows] * total_rotations)
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


def generate_two_arrows_train(num_arrows_pairs, dataset_folder, dataset_name,
                              color_pairs: Optional[List[str]] = None, style_pairs: Optional[List[str]] = None,
                              examples_per_pair: int = 100,
                              resolution=(64, 64)):
    equiv_data = []
    equiv_lbls = []
    equiv_stabilizers = []
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    if style_pairs is None:
        style_pairs = [["simple", "simple"]]
    if color_pairs is None:
        color_pairs = [["tab:red", "tab:red"]]

    for num_pair, arrow_pair in enumerate(num_arrows_pairs):
        for styles in style_pairs:
            for colors in color_pairs:
                for num_example in range(examples_per_pair):
                    print(
                        "Generating image with num arrows {}, style {}, color {}, num example {}".format(
                            arrow_pair, styles, colors, num_example))

                    # Generate image 1
                    c1 = ArrowCanvas(resolution)
                    angle_image1_arrow1 = 2 * np.pi * np.random.random()
                    angle_image1_arrow2 = 2 * np.pi * np.random.random()
                    # Add arrows
                    c1.add_arrows(angle_image1_arrow1, num_arrows=arrow_pair[0], color=colors[0],
                                  start_point=(0.5, 0.5), radius=0.5, style=styles[0])
                    c1.add_arrows(angle_image1_arrow2, num_arrows=arrow_pair[1], color=colors[1],
                                  start_point=(-0.5, -0.5), radius=0.5, style=styles[1])
                    c1.show()
                    # Generate image 2
                    c2 = ArrowCanvas(resolution)
                    angle_image2_arrow1 = 2 * np.pi * np.random.random()
                    angle_image2_arrow2 = 2 * np.pi * np.random.random()
                    # Add arrows
                    c2.add_arrows(angle_image2_arrow1, num_arrows=arrow_pair[0], color=colors[0],
                                  start_point=(0.5, 0.5), radius=0.5, style=styles[0])
                    c2.add_arrows(angle_image2_arrow2, num_arrows=arrow_pair[1], color=colors[1],
                                  start_point=(-0.5, -0.5), radius=0.5, style=styles[1])
                    c2.show()

                    # Get images
                    img1 = c1.numpy_torch_img
                    img2 = c2.numpy_torch_img

                    # Get labels
                    delta_arrow1 = angle_image2_arrow1 - angle_image1_arrow1
                    delta_arrow2 = angle_image2_arrow2 - angle_image1_arrow2

                    # Store data
                    equiv_data.append([img1, img2])
                    equiv_lbls.append([delta_arrow1, delta_arrow2])
                    equiv_stabilizers.append([arrow_pair[0], arrow_pair[1]])

                    plt.close("all")
    equiv_data = np.array(equiv_data)
    equiv_lbls = np.array(equiv_lbls)
    equiv_stabilizers = np.array(equiv_stabilizers)
    print("Equiv data shape", equiv_data.shape)
    print("Equiv lbls shape", equiv_lbls.shape)
    print("Equiv stabilizers shape", equiv_stabilizers.shape)

    np.save(os.path.join(dataset_folder, dataset_name + '_data.npy'), equiv_data)
    np.save(os.path.join(dataset_folder, dataset_name + '_lbls.npy'), equiv_lbls)
    np.save(os.path.join(dataset_folder, dataset_name + '_stabilizers.npy'), equiv_stabilizers)


def generate_two_arrows_eval(num_arrows_pairs, dataset_folder, dataset_name,
                             color_pairs: Optional[List[str]] = None, style_pairs: Optional[List[str]] = None,
                             total_rotations: int = 36,
                             resolution=(64, 64)):
    # if not os.path.exists(dataset_folder):
    #     os.makedirs(dataset_folder)
    if style_pairs is None:
        style_pairs = [["simple", "simple"]]
    if color_pairs is None:
        color_pairs = [["tab:red", "tab:red"]]

    # Regular dataset
    images = []
    stabilizers = []
    labels = []

    for num_pair, arrow_pair in enumerate(num_arrows_pairs):
        for styles in style_pairs:
            for colors in color_pairs:
                images_per_object = []
                labels_per_object = []
                stabilizers_per_object = []
                for num_angle1, angle1 in enumerate(2 * np.pi * np.linspace(0, 1, total_rotations)):
                    for num_angle2, angle2 in enumerate(2 * np.pi * np.linspace(0, 1, total_rotations)):
                        print(
                            "Generating image with num arrows {}, style {}, color {}, num angle1 {} num angle 2 {}".format(
                                arrow_pair, styles, colors, num_angle1, num_angle2))

                        # Generate image 1
                        c1 = ArrowCanvas(resolution)
                        # Add arrows
                        c1.add_arrows(angle1, num_arrows=arrow_pair[0], color=colors[0],
                                      start_point=(0.5, 0.5), radius=0.5, style=styles[0])
                        c1.add_arrows(angle2, num_arrows=arrow_pair[1], color=colors[1],
                                      start_point=(-0.5, -0.5), radius=0.5, style=styles[1])
                        c1.show()

                        # Get images
                        images_per_object.append(c1.numpy_torch_img)
                        labels_per_object.append([angle1, angle2])
                        stabilizers_per_object.append([arrow_pair[0], arrow_pair[1]])
                        plt.close("all")

                images.append(images_per_object)
                stabilizers.append(stabilizers_per_object)
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


if __name__ == "__main__":
    print("Generating dataset")
    # generate_two_arrows_train([(3, 4)], "datasets", "two_arrows", total_rotations=36, examples_per_pair=1)

    # generate_two_arrows_train([(1, 2)], "", "", examples_per_pair=10)
    generate_two_arrows_eval([(1, 2)], "", "", total_rotations=10)
