#!/bin/bash
source /home/TUE/s161416/miniconda3/etc/profile.d/conda.sh
source activate gpytorch
# Sinusoidal data generation
#python3 generate_data.py --n_examples 10000 --dataset_name 1 --n_arrows 1 --dataset sinusoidal
#python3 generate_data.py --n_examples 10000 --dataset_name 2 --n_arrows 2 --dataset sinusoidal
#python3 generate_data.py --n_examples 10000 --dataset_name 3 --n_arrows 3 --dataset sinusoidal
#python3 generate_data.py --n_examples 10000 --dataset_name 3 --n_arrows 5 --dataset sinusoidal
#python3 generate_data.py --n_examples 10000 --dataset_name 7 --n_arrows 7 --dataset sinusoidal
## Arrows data generation
#python3 generate_data.py --n_examples 10000 --dataset_name 1 --n_arrows 1 --dataset rot-arrows
#python3 generate_data.py --n_examples 10000 --dataset_name 2 --n_arrows 2 --dataset rot-arrows
#python3 generate_data.py --n_examples 10000 --dataset_name 3 --n_arrows 3 --dataset rot-arrows
#python3 generate_data.py --n_examples 10000 --dataset_name 5 --n_arrows 5 --dataset rot-arrows
python3 generate_data.py --n_examples 10000 --n_arrows 1 --dataset_name cube --dataset symmetric_solids

#python3 generate_data.py --n_examples 10000 --dataset_name 1 --n_arrows 1 --dataset square_translation
#python3 generate_data.py --n_examples 2000 --dataset_name 1_c --n_arrows 1 --dataset arrows --colors "tab:red" "tab:green" "tab:purple" "tab:orange" "tab:blue"
#python3 generate_data.py --n_examples 2000 --dataset_name 2_c --n_arrows 2 --dataset arrows --colors "tab:red" "tab:green" "tab:purple" "tab:orange" "tab:blue"
#python3 generate_data.py --n_examples 2000 --dataset_name 3_c --n_arrows 3 --dataset arrows --colors "tab:red" "tab:green" "tab:purple" "tab:orange" "tab:blue"
#python3 generate_data.py --n_examples 2000 --dataset_name 5_c --n_arrows 5 --dataset arrows --colors "tab:red" "tab:green" "tab:purple" "tab:orange" "tab:blue"
#python3 generate_data.py --n_examples 2000 --dataset_name 7_c --n_arrows 7 --dataset arrows --colors "tab:red" "tab:green" "tab:purple" "tab:orange" "tab:blue"
