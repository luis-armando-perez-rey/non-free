import argparse

from dataset_generation.simple_sinusoidal import generate_dataset_sinusoidals, generate_dataset_regular_sinusoidals, \
    make_sinusoidal_image
from dataset_generation import image_translation, dsprites_loader, symmetric_solids, rotating_arrows, rotating_mnist, \
    modelnet

parser = argparse.ArgumentParser()
# Dataset
parser.add_argument('--dataset', default='arrows', type=str, help="Dataset")
parser.add_argument('--dataset_name', default='4', type=str, help="Dataset")
parser.add_argument("--n_arrows", nargs="+", dest="n_arrows", type=int, default=[4],
                    help="Number of arrows to generate in dataset")
parser.add_argument("--n_examples", dest="n_examples", default=1000, type=int, help="Number of examples per num arrows")
parser.add_argument('--multicolor', default=False, action='store_true')
parser.add_argument('--split_data', type=str, default="all")
parser.add_argument("--colors", nargs="+", dest="colors", type=str, default=["tab:red"])
parser.add_argument('--styles', nargs='+', default=['simple'], type=str, help="Styles of the arrows")
args = parser.parse_args()


def generate_dataset(dataset):
    if dataset == 'arrows':
        generation_parameters = dict(num_arrows_list=args.n_arrows, color_list=args.colors, style_list=args.styles,
                                     dataset_folder="./data/arrows",
                                     dataset_name=args.dataset_name
                                     )
        if args.multicolor:
            generation_parameters["multicolor"] = True
        if args.split_data == "all":
            rotating_arrows.generate_training_data(**generation_parameters, examples_per_num_arrows=args.n_examples)
            rotating_arrows.generate_eval_data(**generation_parameters)
        elif args.split_data == "train":
            rotating_arrows.generate_training_data(**generation_parameters, examples_per_num_arrows=args.n_examples)
        elif args.split_data == "eval":
            rotating_arrows.generate_eval_data(**generation_parameters)
        else:
            raise ValueError("Unknown split data")

    elif dataset == "double_arrows":

        generation_parameters = dict(num_arrows_pairs=[(args.n_arrows[0], args.n_arrows[1])], color_pairs=None,
                                     style_pairs=None,
                                     dataset_folder="./data/double_arrows",
                                     dataset_name=args.dataset_name
                                     )
        if args.multicolor:
            generation_parameters["multicolor"] = True
        if args.split_data == "all":
            rotating_arrows.generate_two_arrows_train(**generation_parameters, examples_per_pair=args.n_examples)
            rotating_arrows.generate_two_arrows_eval(**generation_parameters)
        elif args.split_data == "train":
            rotating_arrows.generate_two_arrows_train(**generation_parameters, examples_per_pair=args.n_examples)
        elif args.split_data == "eval":
            rotating_arrows.generate_two_arrows_eval(**generation_parameters)
        else:
            raise ValueError("Unknown split data")

    elif dataset == 'sinusoidal':
        generation_parameters = dict(omega_list=args.n_arrows,
                                     dimension=100,
                                     dataset_folder="./data/sinusoidal",
                                     dataset_name=args.dataset_name
                                     )
        generate_dataset_sinusoidals(**generation_parameters, num_examples=args.n_examples)
        generate_dataset_regular_sinusoidals(**generation_parameters, num_angles=36)
    elif dataset == "square_translation":
        image = image_translation.get_square_image()
        image_translation.generate_training_data(image, args.n_examples, "./data/square_translation", args.dataset_name)
        image_translation.generate_eval_data(image, 32, "./data/square_translation", args.dataset_name)
    elif dataset == "dsprites_translation":
        images = dsprites_loader.get_images_shapes()
        image_translation.generate_training_data(images, args.n_examples, "./data/dsprites_translation",
                                                 args.dataset_name)
        image_translation.generate_eval_data(images, 32, "./data/dsprites_translation", args.dataset_name)
    elif dataset == "sinusoidal_translation":
        image = make_sinusoidal_image(omega1=args.n_arrows[0], omega2=args.n_arrows[1])
        image_translation.generate_training_data(image, args.n_examples, "./data/sinusoidal_translation",
                                                 args.dataset_name)
        image_translation.generate_eval_data(image, 32, "./data/sinusoidal_translation", args.dataset_name)
    elif dataset == "rotating_mnist":
        rotating_mnist.generate_training_data("./data/rotating_mnist",
                                              args.dataset_name, args.n_examples)
        rotating_mnist.generate_eval_data("./data/rotating_mnist", args.dataset_name, total_rotations=36)
    elif dataset == "rotating_mnist_stochastic":
        print("Generating stochastic rotating mnist, number of examples {}".format(args.n_examples))
        rotating_mnist.generate_training_data_stochastic("./data/rotating_mnist_stochastic",
                                              args.dataset_name, args.n_examples)
        rotating_mnist.generate_eval_data("./data/rotating_mnist_stochastic", args.dataset_name, total_rotations=36)
    elif dataset == "modelnet":
        assert args.dataset_name in modelnet.render_dictionary.keys(), f"Unknown modelnet object {args.dataset_name}"
        render_details = modelnet.render_dictionary[args.dataset_name]
        modelnet.generate_training_data("./data/modelnet", args.dataset_name,object_name=render_details[0],object_id=render_details[1],examples_per_object=args.n_examples, total_angles=720)
        modelnet.generate_eval_data("./data/modelnet", args.dataset_name,object_name=render_details[0],object_id=render_details[1])
    elif dataset == "symmetric_solids":
        print("NOTICE: In symmetric_solids dataset n_arrows option corresponds to the number of the shape id. E.g. 0 "
              "corresponds to the tetrahedron 1 to the cube, etc. see "
              "https://www.tensorflow.org/datasets/catalog/symmetric_solids for more info. ")
        symmetric_solids.generate_training_data("./data/symmetric_solids", "./data/symmetric_solids",
                                                dataset_name=args.dataset_name,
                                                shape_id=args.n_arrows[0], num_pairs=args.n_examples)
    else:
        raise ValueError(f"Dataset {dataset} not supported")


if __name__ == "__main__":
    generate_dataset(args.dataset)
