#!/bin/bash
# source /home/TUE/s161416/miniconda3/etc/profile.d/conda.sh
# source activate gpytorch
# Sinusoidal data generation
#python3 generate_data.py --n_examples 10000 --dataset_name 1 --n_arrows 1 --dataset sinusoidal
#python3 generate_data.py --n_examples 10000 --dataset_name 2 --n_arrows 2 --dataset sinusoidal
#python3 generate_data.py --n_examples 10000 --dataset_name 3 --n_arrows 3 --dataset sinusoidal
#python3 generate_data.py --n_examples 10000 --dataset_name 5 --n_arrows 5 --dataset sinusoidal
#python3 generate_data.py --n_examples 10000 --dataset_name 7 --n_arrows 7 --dataset sinusoidal
## Arrows data generation
#python3 generate_data.py --n_examples 10000 --dataset_name 1 --n_arrows 1 --dataset rot-arrows
#python3 generate_data.py --n_examples 10000 --dataset_name 2 --n_arrows 2 --dataset rot-arrows
#python3 generate_data.py --n_examples 10000 --dataset_name 3 --n_arrows 3 --dataset rot-arrows
#python3 generate_data.py --n_examples 5000 --dataset_name 5 --n_arrows 5 --dataset arrows
#python3 generate_data.py --n_examples 5000 --dataset_name 4 --n_arrows 4 --dataset arrows
#python3 generate_data.py --n_examples 5000 --dataset_name 5 --n_arrows 5 --dataset arrows
#python3 generate_data.py --n_examples 5000 --dataset_name 6 --n_arrows 6 --dataset arrows
#python generate_data.py --n_examples 4500 --dataset_name 4 --n_arrows 4 --dataset arrows
#python generate_data.py --n_examples 4500 --dataset_name 5 --n_arrows 5 --dataset arrows

#python generate_data.py --n_examples 5000 --dataset_name 1 --n_arrows 1 --dataset arrows
#python generate_data.py --n_examples 5000 --dataset_name 2 --n_arrows 2 --dataset arrows
#python generate_data.py --n_examples 5000 --dataset_name 3 --n_arrows 3 --dataset arrows
#python generate_data.py --n_examples 5000 --dataset_name 4 --n_arrows 4 --dataset arrows
#python generate_data.py --n_examples 5000 --dataset_name 5 --n_arrows 5 --dataset arrows
#
#python generate_data.py --n_examples 250 --dataset_name 1_val --n_arrows 1 --dataset arrows
#python generate_data.py --n_examples 250 --dataset_name 2_val --n_arrows 2 --dataset arrows
#python generate_data.py --n_examples 250 --dataset_name 3_val --n_arrows 3 --dataset arrows
#python generate_data.py --n_examples 250 --dataset_name 4_val --n_arrows 4 --dataset arrows
#python generate_data.py --n_examples 250 --dataset_name 5_val --n_arrows 5 --dataset arrows

# Pairs of arrows
#python3 generate_data.py --n_examples 2000 --dataset_name 2_3 --n_arrows 2 3 --dataset double_arrows
#python3 generate_data.py --n_examples 250 --dataset_name 2_3_val --n_arrows 2 3 --dataset double_arrows
#python3 generate_data.py --n_examples 2000 --dataset_name 3_5 --n_arrows 3 5 --dataset double_arrows
#python3 generate_data.py --n_examples 250 --dataset_name 3_5_val --n_arrows 3 5 --dataset double_arrows
#python3 generate_data.py --n_examples 1 --dataset_name test --dataset rotating_mnist
#python3 generate_data.py --n_examples 1000 --dataset_name stochastic_mnist --dataset rotating_mnist_stochastic


# Modelnet
python3 generate_data.py --n_examples 2500 --dataset_name airplane_0 --dataset modelnet
python3 generate_data.py --n_examples 250 --dataset_name airplane_0_val --dataset modelnet
python3 generate_data.py --n_examples 2500 --dataset_name bathtub_0 --dataset modelnet
python3 generate_data.py --n_examples 250 --dataset_name bathtub_0_val --dataset modelnet
python3 generate_data.py --n_examples 2500 --dataset_name stool_0 --dataset modelnet
python3 generate_data.py --n_examples 250 --dataset_name stool_0_val --dataset modelnet
python3 generate_data.py --n_examples 2500 --dataset_name bottle_0 --dataset modelnet
python3 generate_data.py --n_examples 250 --dataset_name bottle_0_val --dataset modelnet
python3 generate_data.py --n_examples 2500 --dataset_name chair_0 --dataset modelnet
python3 generate_data.py --n_examples 250 --dataset_name chair_0_val --dataset modelnet
python3 generate_data.py --n_examples 2500 --dataset_name lamp_0 --dataset modelnet
python3 generate_data.py --n_examples 250 --dataset_name lamp_0_val --dataset modelnet
python3 generate_data.py --n_examples 2500 --dataset_name bookshelf_0 --dataset modelnet
python3 generate_data.py --n_examples 250 --dataset_name bookshelf_0_val --dataset modelnet




# Modelnet
#python3 generate_data.py --n_examples 2500 --dataset_name airplane_0 --dataset modelnet
#python3 generate_data.py --n_examples 250 --dataset_name airplane_0_val --dataset modelnet
#python3 generate_data.py --n_examples 2500 --dataset_name bottle_0 --dataset modelnet
#python3 generate_data.py --n_examples 250 --dataset_name bottle_0_val --dataset modelnet
#python3 generate_data.py --n_examples 2500 --dataset_name chair_0 --dataset modelnet
#python3 generate_data.py --n_examples 250 --dataset_name chair_0_val --dataset modelnet


#python3 generate_data.py --n_examples 1000 --dataset_name stool_0 --dataset modelnet
#python3 generate_data.py --n_examples 2000 --dataset_name bookshelf_0 --dataset modelnet
#python3 generate_data.py --n_examples 2000 --dataset_name bottle_0 --dataset modelnet
#python3 generate_data.py --n_examples 2000 --dataset_name stool_2 --dataset modelnet
#python3 generate_data.py --n_examples 1000 --dataset_name stool_1 --dataset modelnet

#python3 generate_data.py --n_examples 10000 --n_arrows 5 --dataset_name marked_tetrahedron --dataset symmetric_solids
#python3 generate_data.py --n_examples 10000 --n_arrows 0 --dataset_name tetrahedron --dataset symmetric_solids

#python3 generate_data.py --n_examples 10000 --dataset_name 1 --n_arrows 1 --dataset square_translation
#python3 generate_data.py --n_examples 20000 --dataset_name 1_1 --n_arrows 1 1 --dataset sinusoidal_translation
#python3 generate_data.py --n_examples 20000 --dataset_name 2_1 --n_arrows 2 1 --dataset sinusoidal_translation
#python3 generate_data.py --n_examples 20000 --dataset_name 1_2 --n_arrows 1 2 --dataset sinusoidal_translation
#python3 generate_data.py --n_examples 20000 --dataset_name 2_2 --n_arrows 2 2 --dataset sinusoidal_translation
#python3 generate_data.py --n_examples 20000 --dataset_name 3_3 --n_arrows 3 3 --dataset sinusoidal_translation
#python3 generate_data.py --n_examples 2000 --dataset_name 1_c --n_arrows 1 --dataset arrows --colors "tab:red" "tab:green" "tab:purple" "tab:orange" "tab:blue"
#python3 generate_data.py --n_examples 2000 --dataset_name 2_c --n_arrows 2 --dataset arrows --colors "tab:red" "tab:green" "tab:purple" "tab:orange" "tab:blue"
#python3 generate_data.py --n_examples 2000 --dataset_name 3_c --n_arrows 3 --dataset arrows --colors "tab:red" "tab:green" "tab:purple" "tab:orange" "tab:blue"
#python3 generate_data.py --n_examples 2000 --dataset_name 4_c --n_arrows 4 --dataset arrows --colors "tab:red" "tab:green" "tab:purple" "tab:orange" "tab:blue"
#python3 generate_data.py --n_examples 2000 --dataset_name 5_c --n_arrows 5 --dataset arrows --colors "tab:red" "tab:green" "tab:purple" "tab:orange" "tab:blue"
#
#python3 generate_data.py --n_examples 200 --dataset_name 1_c_val --n_arrows 1 --dataset arrows --colors "tab:red" "tab:green" "tab:purple" "tab:orange" "tab:blue"
#python3 generate_data.py --n_examples 200 --dataset_name 2_c_val --n_arrows 2 --dataset arrows --colors "tab:red" "tab:green" "tab:purple" "tab:orange" "tab:blue"
#python3 generate_data.py --n_examples 200 --dataset_name 3_c_val --n_arrows 3 --dataset arrows --colors "tab:red" "tab:green" "tab:purple" "tab:orange" "tab:blue"
#python3 generate_data.py --n_examples 200 --dataset_name 4_c_val --n_arrows 4 --dataset arrows --colors "tab:red" "tab:green" "tab:purple" "tab:orange" "tab:blue"
#python3 generate_data.py --n_examples 200 --dataset_name 5_c_val --n_arrows 5 --dataset arrows --colors "tab:red" "tab:green" "tab:purple" "tab:orange" "tab:blue"

# Discrete arrows
#python3 generate_data.py --n_examples 1500 --dataset_name 1d --n_arrows 1 --dataset discrete_arrows
#python3 generate_data.py --n_examples 1500 --dataset_name 2d --n_arrows 2 --dataset discrete_arrows
#python3 generate_data.py --n_examples 1500 --dataset_name 3d --n_arrows 3 --dataset discrete_arrows
#python3 generate_data.py --n_examples 1500 --dataset_name 4d --n_arrows 4 --dataset discrete_arrows
#python3 generate_data.py --n_examples 1500 --dataset_name 5d --n_arrows 5 --dataset discrete_arrows
#
#python3 generate_data.py --n_examples 100 --dataset_name 1d_val --n_arrows 1 --dataset discrete_arrows
#python3 generate_data.py --n_examples 100 --dataset_name 2d_val --n_arrows 2 --dataset discrete_arrows
#python3 generate_data.py --n_examples 100 --dataset_name 3d_val --n_arrows 3 --dataset discrete_arrows
#python3 generate_data.py --n_examples 100 --dataset_name 4d_val --n_arrows 4 --dataset discrete_arrows
#python3 generate_data.py --n_examples 100 --dataset_name 5d_val --n_arrows 5 --dataset discrete_arrows

# Discrete quessard arrows
#python3 generate_data.py --n_examples 60 --dataset_name 1q --n_arrows 1 --dataset discrete_quessard_arrows
#python3 generate_data.py --n_examples 60 --dataset_name 1_2q --n_arrows 1 2 --dataset discrete_quessard_arrows
#python3 generate_data.py --n_examples 60 --dataset_name 1_2_3_4_5q --n_arrows 1 2 3 4 5 --dataset discrete_quessard_arrows



# Modelnet quessard
#python3 generate_data.py --n_examples 0 --dataset_name airplane_0-chair_0 --n_arrows 1 --dataset modelnet_quessard

# Symmetric solids
#python3 generate_data.py --n_examples 7500 --n_arrows 0 --dataset_name tetrahedron --dataset symmetric_solids
#python3 generate_data.py --n_examples 750 --n_arrows 0 --dataset_name tetrahedron_val --dataset symmetric_solids
#python3 generate_data.py --n_examples 7500 --n_arrows 1 --dataset_name cube --dataset symmetric_solids
#python3 generate_data.py --n_examples 750 --n_arrows 1 --dataset_name cube_val --dataset symmetric_solids
#
#python3 generate_data.py --n_examples 7500 --n_arrows 2 --dataset_name icosahedron --dataset symmetric_solids
#python3 generate_data.py --n_examples 750 --n_arrows 2 --dataset_name icosahedron_val --dataset symmetric_solids

#python3 generate_data.py --n_examples 7500 --n_arrows 5 --dataset_name marked_tetrahedron --dataset symmetric_solids
#python3 generate_data.py --n_examples 750 --n_arrows 5 --dataset_name marked_tetrahedron_val --dataset symmetric_solids