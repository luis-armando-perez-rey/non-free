#!/bin/sh

python3 generate_data.py --n_examples 10000 --dataset_name 34 --n_arrows 3 4
python3 generate_data.py --n_examples 10000 --dataset_name 3 --n_arrows 3
python3 generate_data.py --n_examples 10000 --dataset_name 4 --n_arrows 4
