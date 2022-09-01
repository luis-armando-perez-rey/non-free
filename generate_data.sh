#!/bin/sh
python3 generate_data.py --n_examples 10000 --dataset_name 1 --n_arrows 1 --dataset sinusoidal
python3 generate_data.py --n_examples 10000 --dataset_name 2 --n_arrows 2 --dataset sinusoidal
python3 generate_data.py --n_examples 10000 --dataset_name 3 --n_arrows 3 --dataset sinusoidal
python3 generate_data.py --n_examples 10000 --dataset_name 5 --n_arrows 5 --dataset sinusoidal
python3 generate_data.py --n_examples 10000 --dataset_name 7 --n_arrows 7 --dataset sinusoidal
#python3 generate_data.py --n_examples 10000 --dataset_name 6 --n_arrows 1
#python3 generate_data.py --n_examples 10000 --dataset_name 4_m --n_arrows 4 --multicolor

