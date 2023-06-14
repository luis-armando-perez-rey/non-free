import argparse
import os
from dataset_generation.simple_sinusoidal import generate_dataset_sinusoidals, generate_dataset_regular_sinusoidals, \
    make_sinusoidal_image
from dataset_generation import image_translation, dsprites_loader, symmetric_solids, rotating_arrows, rotating_mnist, \
    modelnet, modelnet_so3, modelnet_regular


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

        generate_eval_data_parameters = dict(num_arrows_list=args.n_arrows, color_list=args.colors,
                                               style_list=args.styles,
                                               dataset_folder="./data/arrows",
                                               dataset_name=args.dataset_name
                                               )
        if args.multicolor:
            generate_eval_data_parameters["multicolor"] = True
        generate_train_data_parameters = generate_eval_data_parameters.copy()
        generate_train_data_parameters["examples_per_num_arrows"] = args.n_examples

        train_data_function = rotating_arrows.generate_training_data
        eval_data_function = rotating_arrows.generate_eval_data

    elif dataset == "consecutive_arrows":
        generate_eval_data_parameters = dict(num_arrows_list=args.n_arrows, color_list=args.colors,
                                               style_list=args.styles,
                                               dataset_folder="./data/consecutive_arrows",
                                               dataset_name=args.dataset_name,
                                               )
        generate_train_data_parameters = generate_eval_data_parameters.copy()
        generate_train_data_parameters["examples_per_num_arrows"] = args.n_examples
        generate_train_data_parameters["type_pairs"] = "consecutive"
        generate_train_data_parameters["num_discrete_angles"] = args.n_examples

        train_data_function = rotating_arrows.generate_training_data
        eval_data_function = rotating_arrows.generate_eval_data
    elif dataset == "consecutive_random_arrows":
        generate_eval_data_parameters = dict(num_arrows_list=args.n_arrows, color_list=args.colors,
                                               style_list=args.styles,
                                               dataset_folder="./data/consecutive_random_arrows",
                                               dataset_name=args.dataset_name,
                                               )
        generate_train_data_parameters = generate_eval_data_parameters.copy()
        generate_train_data_parameters["examples_per_num_arrows"] = args.n_examples
        generate_train_data_parameters["type_pairs"] = "consecutive_random"
        generate_train_data_parameters["num_discrete_angles"] = args.n_examples

        train_data_function = rotating_arrows.generate_training_data
        eval_data_function = rotating_arrows.generate_eval_data
    elif dataset == "discrete_arrows":
        generate_eval_data_parameters = dict(num_arrows_list=args.n_arrows, color_list=args.colors,
                                               style_list=args.styles,
                                               dataset_folder="./data/discrete_arrows",
                                               dataset_name=args.dataset_name,
                                               )
        generate_train_data_parameters = generate_eval_data_parameters.copy()
        generate_train_data_parameters["examples_per_num_arrows"] = args.n_examples
        generate_train_data_parameters["type_pairs"] = "discrete_arrows"

        train_data_function = rotating_arrows.generate_training_data
        eval_data_function = rotating_arrows.generate_eval_data

    elif dataset == "double_arrows":

        generate_eval_data_parameters = dict(num_arrows_pairs=[(args.n_arrows[0], args.n_arrows[1])],
                                               color_pairs=None,
                                               style_pairs=None,
                                               dataset_folder="./data/double_arrows",
                                               dataset_name=args.dataset_name
                                               )
        if args.multicolor:
            generate_eval_data_parameters["multicolor"] = True
        generate_train_data_parameters = generate_eval_data_parameters.copy()
        generate_train_data_parameters["examples_per_pair"] = args.n_examples

        train_data_function = rotating_arrows.generate_two_arrows_train
        eval_data_function = rotating_arrows.generate_two_arrows_eval


    elif dataset == 'sinusoidal':
        generate_eval_data_parameters = dict(omega_list=args.n_arrows,
                                               dimension=100,
                                               dataset_folder="./data/sinusoidal",
                                               dataset_name=args.dataset_name
                                               )
        generate_train_data_parameters = generate_eval_data_parameters.copy()
        generate_train_data_parameters["num_examples"] = args.n_examples
        generate_eval_data_parameters["num_angles"] = 36

        train_data_function = generate_dataset_sinusoidals
        eval_data_function = generate_dataset_regular_sinusoidals

    elif dataset == "square_translation":
        image = image_translation.get_square_image()
        generate_eval_data_parameters = dict(image=image,
                                               n_datapoints=32,
                                               dataset_folder="./data/square_translation",
                                               dataset_name=args.dataset_name)
        generate_train_data_parameters = generate_eval_data_parameters.copy()
        generate_train_data_parameters["n_datapoints"] = args.n_examples

        train_data_function = image_translation.generate_training_data
        eval_data_function = image_translation.generate_eval_data

    elif dataset == "dsprites_translation":
        generate_eval_data_parameters = dict(images=dsprites_loader.get_images_shapes(),
                                               n_datapoints=32,
                                               dataset_folder="./data/dsprites_translation",
                                               dataset_name=args.dataset_name)
        generate_train_data_parameters = generate_eval_data_parameters.copy()
        generate_train_data_parameters["n_datapoints"] = args.n_examples

        train_data_function = image_translation.generate_training_data
        eval_data_function = image_translation.generate_eval_data

    elif dataset == "sinusoidal_translation":
        generate_eval_data_parameters = dict(
            images=make_sinusoidal_image(omega1=args.n_arrows[0], omega2=args.n_arrows[1]),
            n_datapoints=32,
            dataset_folder="./data/sinusoidal_translation",
            dataset_name=args.dataset_name)
        generate_train_data_parameters = generate_eval_data_parameters.copy()
        generate_train_data_parameters["n_datapoints"] = args.n_examples

        train_data_function = image_translation.generate_training_data
        eval_data_function = image_translation.generate_eval_data
    elif dataset == "rotating_mnist":
        # TODO: Fix this I am working here
        generate_eval_data_parameters = dict(dataset_folder="./data/rotating_mnist",
                                               dataset_name=args.dataset_name,
                                               total_rotations=36)
        generate_train_data_parameters = generate_eval_data_parameters.copy()
        generate_train_data_parameters["examples_per_digit"] = args.n_examples
        generate_train_data_parameters.pop("total_rotations")

        train_data_function = rotating_mnist.generate_training_data
        eval_data_function = rotating_mnist.generate_eval_data

    elif dataset == "rotating_mnist_stochastic":
        print("Generating stochastic rotating mnist, number of examples {}".format(args.n_examples))
        generate_eval_data_parameters = dict(dataset_folder="./data/rotating_mnist_stochastic",
                                               dataset_name=args.dataset_name,
                                               total_rotations=36)
        generate_train_data_parameters = generate_eval_data_parameters.copy()
        generate_train_data_parameters["examples_per_digit"] = args.n_examples
        generate_train_data_parameters.pop("total_rotations")

        train_data_function = rotating_mnist.generate_training_data_stochastic
        eval_data_function = rotating_mnist.generate_eval_data

    elif dataset == "modelnet":
        assert args.dataset_name in modelnet.render_dictionary.keys(), f"Unknown modelnet object {args.dataset_name}"
        render_details = modelnet.render_dictionary[args.dataset_name]

        generate_eval_data_parameters = dict(dataset_folder="./data/modelnet",
                                               dataset_name=args.dataset_name,
                                               object_name=render_details[0],
                                               object_id=render_details[1],
                                               split="train",
                                               total_angles=720)
        generate_train_data_parameters = generate_eval_data_parameters.copy()
        generate_train_data_parameters["examples_per_object"] = args.n_examples
        generate_train_data_parameters["total_angles"] = 720

        train_data_function = modelnet.generate_training_data
        eval_data_function = modelnet.generate_eval_data
    elif dataset == "modelnet_regular":
        object_type = args.dataset_name.split("-")[0]
        assert object_type in modelnet_regular.AVAILABLE_OBJECTS, f"Unknown modelnet object {args.dataset_name}"
        # Define data generation parameters
        generate_eval_data_parameters = dict(render_folder="/data/active_views",
                                             dataset_folder=f"./data/{dataset}",
                                             dataset_name=args.dataset_name,
                                             object_type=object_type,
                                             split="train")
        generate_train_data_parameters = generate_eval_data_parameters.copy()
        generate_train_data_parameters["examples_per_object"] = args.n_examples

        train_data_function = modelnet_regular.generate_training_data
        eval_data_function = modelnet_regular.generate_eval_data

    elif dataset == "modelnet_regular_pairs":
        object_type = args.dataset_name.split("-")[0]
        assert object_type in modelnet_regular.AVAILABLE_OBJECTS, f"Unknown modelnet object {args.dataset_name}"

        generate_eval_data_parameters = dict(render_folder="/data/volume_2/data/active_views",
                                                dataset_folder=f"./data/{dataset}",
                                                dataset_name=args.dataset_name,
                                                object_type=object_type,
                                                split="train")
        generate_train_data_parameters = generate_eval_data_parameters.copy()
        generate_train_data_parameters["examples_per_object"] = args.n_examples
        generate_train_data_parameters["use_random_choice"] = False

        train_data_function = modelnet_regular.generate_training_data
        eval_data_function = modelnet_regular.generate_eval_data

    elif dataset == "modelnet_regular_pairs_test":
        object_type = args.dataset_name.split("-")[0]
        assert object_type in modelnet_regular.AVAILABLE_OBJECTS, f"Unknown modelnet object {args.dataset_name}"
        generate_eval_data_parameters = dict(render_folder="/data/volume_2/data/active_views",
                                             dataset_folder=f"./data/{dataset}",
                                             dataset_name=args.dataset_name,
                                             object_type=object_type,
                                             split="test")
        generate_train_data_parameters = generate_eval_data_parameters.copy()
        generate_train_data_parameters["examples_per_object"] = args.n_examples
        generate_train_data_parameters["use_random_choice"] = False

        train_data_function = modelnet_regular.generate_training_data
        eval_data_function = modelnet_regular.generate_eval_data
    elif dataset == "modelnet_regular_pairs_test0":
        object_type = args.dataset_name.split("-")[0]
        assert object_type in modelnet_regular.AVAILABLE_OBJECTS, f"Unknown modelnet object {args.dataset_name}"
        generate_eval_data_parameters = dict(render_folder="/data/volume_2/data/active_views",
                                             dataset_folder=f"./data/{dataset}",
                                             dataset_name=args.dataset_name,
                                             object_type=object_type,
                                             split="test")
        generate_train_data_parameters = generate_eval_data_parameters.copy()
        generate_train_data_parameters["examples_per_object"] = args.n_examples
        generate_train_data_parameters["use_random_choice"] = False

        train_data_function = modelnet_regular.generate_training_data
        eval_data_function = modelnet_regular.generate_eval_data
    elif dataset == "modelnet_regular0":
        object_type = args.dataset_name.split("-")[0]
        assert object_type in modelnet_regular.AVAILABLE_OBJECTS, f"Unknown modelnet object {args.dataset_name}"
        generate_eval_data_parameters = dict(render_folder="/data/volume_2/data/active_views",
                                             dataset_folder=f"./data/{dataset}",
                                             dataset_name=args.dataset_name,
                                             object_type=object_type,
                                             split="train")
        generate_train_data_parameters = generate_eval_data_parameters.copy()
        generate_train_data_parameters["examples_per_object"] = args.n_examples
        generate_train_data_parameters["use_random_choice"] = True
        generate_train_data_parameters["use_random_initial"] = False

        train_data_function = modelnet_regular.generate_training_data
        eval_data_function = modelnet_regular.generate_eval_data

    elif dataset == "modelnet_regular_pairs0":
        object_type = args.dataset_name.split("-")[0]
        assert object_type in modelnet_regular.AVAILABLE_OBJECTS, f"Unknown modelnet object {args.dataset_name}"
        generate_eval_data_parameters = dict(render_folder="/data/volume_2/data/active_views",
                                             dataset_folder=f"./data/{dataset}",
                                             dataset_name=args.dataset_name,
                                             object_type=object_type,
                                             split="train")
        generate_train_data_parameters = generate_eval_data_parameters.copy()
        generate_train_data_parameters["examples_per_object"] = args.n_examples
        generate_train_data_parameters["use_random_choice"] = False
        generate_train_data_parameters["use_random_initial"] = False

        train_data_function = modelnet_regular.generate_training_data
        eval_data_function = modelnet_regular.generate_eval_data

    elif dataset == "modelnetso3":
        os.makedirs("./data/" + dataset, exist_ok=True)
        generate_eval_data_parameters = dict(dataset_name=f"./data/{dataset}",
                                             load_folder="./data/modelnet_renders",
                                                save_folder="./data/modelnetso3")
        generate_train_data_parameters = generate_eval_data_parameters.copy()

        train_data_function = modelnet_so3.generate_data
        eval_data_function = modelnet_so3.generate_data

    elif dataset == "symmetric_solids":
        print("NOTICE: In symmetric_solids dataset n_arrows option corresponds to the number of the shape id. E.g. 0 "
              "corresponds to the tetrahedron 1 to the cube, etc. see "
              "https://www.tensorflow.org/datasets/catalog/symmetric_solids for more info. ")
        generate_eval_data_parameters = dict(load_folder="./data/symmetric_solids",
                                             save_folder="./data/symmetric_solids",
                                                dataset_name=args.dataset_name,
                                                shape_id=args.n_arrows[0], num_pairs=args.n_examples)
        generate_train_data_parameters = generate_eval_data_parameters.copy()

        train_data_function = symmetric_solids.generate_training_data
        eval_data_function = symmetric_solids.generate_training_data

    else:
        generate_train_data_parameters = dict()
        generate_eval_data_parameters = dict()
        raise ValueError(f"Dataset {dataset} not supported")

    if args.split_data == "all":
        train_data_function(**generate_train_data_parameters)
        eval_data_function(**generate_eval_data_parameters)
    elif args.split_data == "train":
        train_data_function(**generate_train_data_parameters)
    elif args.split_data == "eval":
        eval_data_function(**generate_eval_data_parameters)


if __name__ == "__main__":
    generate_dataset(args.dataset)
