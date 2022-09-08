#!/bin/bash
source /home/TUE/s161416/miniconda3/etc/profile.d/conda.sh
source activate gpytorch
# Sinusoidal data generation
python3 generate_data.py --n_examples 10000 --dataset_name 1 --n_arrows 1 --dataset sinusoidal
python3 generate_data.py --n_examples 10000 --dataset_name 2 --n_arrows 2 --dataset sinusoidal
python3 generate_data.py --n_examples 10000 --dataset_name 3 --n_arrows 3 --dataset sinusoidal
python3 generate_data.py --n_examples 10000 --dataset_name 3 --n_arrows 5 --dataset sinusoidal
python3 generate_data.py --n_examples 10000 --dataset_name 7 --n_arrows 7 --dataset sinusoidal
# Arrows data generation
python3 generate_data.py --n_examples 10000 --dataset_name 1 --n_arrows 1 --dataset rot-arrows
python3 generate_data.py --n_examples 10000 --dataset_name 2 --n_arrows 2 --dataset rot-arrows
python3 generate_data.py --n_examples 10000 --dataset_name 3 --n_arrows 3 --dataset rot-arrows
python3 generate_data.py --n_examples 10000 --dataset_name 5 --n_arrows 5 --dataset rot-arrows
python3 generate_data.py --n_examples 10000 --dataset_name 7 --n_arrows 7 --dataset rot-arrows

