#!/bin/sh
python3 generate_data.py --n_examples 10000 --dataset_name 3 --n_arrows 3
python3 generate_data.py --n_examples 10000 --dataset_name 4_m --n_arrows 4 --multicolor

