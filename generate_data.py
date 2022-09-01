import argparse
from dataset_generation.rotating_arrows import generate_training_data, generate_eval_data
from dataset_generation.simple_sinusoidal import generate_dataset_sinusoidals, generate_dataset_regular_sinusoidals
parser = argparse.ArgumentParser()
# Dataset
parser.add_argument('--dataset', default='rot-arrows', type=str, help="Dataset")
parser.add_argument('--dataset_name', default='4', type=str, help="Dataset")
parser.add_argument("--n_arrows", nargs="+", dest="n_arrows", type=int, default=[4],
                        help="Number of arrows to generate in dataset")
parser.add_argument("--n_examples", dest="n_examples", default=1000, type = int, help="Number of examples per num arrows")
parser.add_argument('--multicolor', default=False, action='store_true')
args = parser.parse_args()


def generate_dataset(dataset):
    if dataset == 'rot-arrows':
        generation_parameters = dict(num_arrows_list=args.n_arrows, color_list=["tab:red"], style_list=["simple"],
                                     dataset_folder="./data/arrows",
                                     dataset_name=args.dataset_name
                                     )
        if args.multicolor:
            generation_parameters["multicolor"] = True
        generate_training_data(**generation_parameters, examples_per_num_arrows=args.n_examples)
        generate_eval_data(**generation_parameters)
    elif dataset == 'sinusoidal':
        generation_parameters = dict(omega_list=args.n_arrows,
                                     dimension=100,
                                     dataset_folder="./data/sinusoidal",
                                     dataset_name=args.dataset_name
                                     )
        generate_dataset_sinusoidals(**generation_parameters, num_examples=args.n_examples)
        generate_dataset_regular_sinusoidals(**generation_parameters, num_angles=36)


if __name__ == "__main__":
    generate_dataset(args.dataset)