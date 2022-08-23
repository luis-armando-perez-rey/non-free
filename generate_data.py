import argparse
from dataset_generation.rotating_arrows import generate_training_data, generate_eval_data

parser = argparse.ArgumentParser()
# Dataset
parser.add_argument('--dataset', default='rot-arrows', type=str, help="Dataset")
parser.add_argument('--dataset_name', default='4', type=str, help="Dataset")
parser.add_argument("--n_arrows", nargs="+", dest="n_arrows", type=int, default=[4],
                        help="Number of arrows to generate in dataset")
parser.add_argument("--n_examples", dest="n_examples", default=1000, type = int, help="Number of examples per num arrows")
args = parser.parse_args()


def generate_dataset(dataset):
    if dataset == 'rot-arrows':
        generation_parameters = dict(num_arrows_list=args.n_arrows, color_list=["tab:red"], style_list=["simple"],
                                     dataset_folder="./data/arrows",
                                     dataset_name=args.dataset_name
                                     )
        generate_training_data(**generation_parameters, examples_per_num_arrows=args.n_examples)
        generate_eval_data(**generation_parameters)


if __name__ == "__main__":
    generate_dataset(args.dataset)
