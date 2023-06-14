#!/bin/bash
# Store argument in dataset_name variable
dataset_name=$1
# Run code based on dataset_name variable value
case $dataset_name in
    "arrows")
    python generate_data.py --n_examples 5000 --dataset_name 1 --n_arrows 1 --dataset arrows
    python generate_data.py --n_examples 5000 --dataset_name 2 --n_arrows 2 --dataset arrows
    python generate_data.py --n_examples 5000 --dataset_name 3 --n_arrows 3 --dataset arrows
    python generate_data.py --n_examples 5000 --dataset_name 4 --n_arrows 4 --dataset arrows
    python generate_data.py --n_examples 5000 --dataset_name 5 --n_arrows 5 --dataset arrows

    python generate_data.py --n_examples 250 --dataset_name 1_val --n_arrows 1 --dataset arrows
    python generate_data.py --n_examples 250 --dataset_name 2_val --n_arrows 2 --dataset arrows
    python generate_data.py --n_examples 250 --dataset_name 3_val --n_arrows 3 --dataset arrows
    python generate_data.py --n_examples 250 --dataset_name 4_val --n_arrows 4 --dataset arrows
    python generate_data.py --n_examples 250 --dataset_name 5_val --n_arrows 5 --dataset arrows
    ;;
    "arrows_colors")
    # Colored arrows
    python3 generate_data.py --n_examples 2000 --dataset_name 1_c --n_arrows 1 --dataset arrows --colors "tab:red" "tab:green" "tab:purple" "tab:orange" "tab:blue"
    python3 generate_data.py --n_examples 2000 --dataset_name 2_c --n_arrows 2 --dataset arrows --colors "tab:red" "tab:green" "tab:purple" "tab:orange" "tab:blue"
    python3 generate_data.py --n_examples 2000 --dataset_name 3_c --n_arrows 3 --dataset arrows --colors "tab:red" "tab:green" "tab:purple" "tab:orange" "tab:blue"
    python3 generate_data.py --n_examples 2000 --dataset_name 4_c --n_arrows 4 --dataset arrows --colors "tab:red" "tab:green" "tab:purple" "tab:orange" "tab:blue"
    python3 generate_data.py --n_examples 2000 --dataset_name 5_c --n_arrows 5 --dataset arrows --colors "tab:red" "tab:green" "tab:purple" "tab:orange" "tab:blue"

    python3 generate_data.py --n_examples 200 --dataset_name 1_c_val --n_arrows 1 --dataset arrows --colors "tab:red" "tab:green" "tab:purple" "tab:orange" "tab:blue"
    python3 generate_data.py --n_examples 200 --dataset_name 2_c_val --n_arrows 2 --dataset arrows --colors "tab:red" "tab:green" "tab:purple" "tab:orange" "tab:blue"
    python3 generate_data.py --n_examples 200 --dataset_name 3_c_val --n_arrows 3 --dataset arrows --colors "tab:red" "tab:green" "tab:purple" "tab:orange" "tab:blue"
    python3 generate_data.py --n_examples 200 --dataset_name 4_c_val --n_arrows 4 --dataset arrows --colors "tab:red" "tab:green" "tab:purple" "tab:orange" "tab:blue"
    python3 generate_data.py --n_examples 200 --dataset_name 5_c_val --n_arrows 5 --dataset arrows --colors "tab:red" "tab:green" "tab:purple" "tab:orange" "tab:blue"
    ;;
  "modelnet")
  # Modelnet
  python3 generate_data.py --n_examples 2500 --dataset_name airplane_0 --dataset modelnet
  python3 generate_data.py --n_examples 250 --dataset_name airplane_0_val --dataset modelnet
  python3 generate_data.py --n_examples 2500 --dataset_name chair_0 --dataset modelnet
  python3 generate_data.py --n_examples 250 --dataset_name chair_0_val --dataset modelnet
  python3 generate_data.py --n_examples 2500 --dataset_name stool_0 --dataset modelnet
  python3 generate_data.py --n_examples 250 --dataset_name stool_0_val --dataset modelnet
  python3 generate_data.py --n_examples 2500 --dataset_name lamp_0 --dataset modelnet
  python3 generate_data.py --n_examples 250 --dataset_name lamp_0_val --dataset modelnet
  python3 generate_data.py --n_examples 2500 --dataset_name bathtub_0 --dataset modelnet
  python3 generate_data.py --n_examples 250 --dataset_name bathtub_0_val --dataset modelnet
  ;;
  "symmetric_solids")
  # Symmetric solids
  python3 generate_data.py --n_examples 7500 --n_arrows 0 --dataset_name tetrahedron --dataset symmetric_solids
  python3 generate_data.py --n_examples 750 --n_arrows 0 --dataset_name tetrahedron_val --dataset symmetric_solids
  python3 generate_data.py --n_examples 7500 --n_arrows 1 --dataset_name cube --dataset symmetric_solids
  python3 generate_data.py --n_examples 750 --n_arrows 1 --dataset_name cube_val --dataset symmetric_solids
  python3 generate_data.py --n_examples 7500 --n_arrows 2 --dataset_name icosahedron --dataset symmetric_solids
  python3 generate_data.py --n_examples 750 --n_arrows 2 --dataset_name icosahedron_val --dataset symmetric_solids
  ;;
  "double_arrows")
  # Double arrows
  python3 generate_data.py --n_examples 2000 --dataset_name 2_3 --n_arrows 2 3 --dataset double_arrows
  python3 generate_data.py --n_examples 2000 --dataset_name 3_5 --n_arrows 3 5 --dataset double_arrows
  python3 generate_data.py --n_examples 200 --dataset_name 2_3_val --n_arrows 2 3 --dataset double_arrows
  python3 generate_data.py --n_examples 200 --dataset_name 3_5_val --n_arrows 3 5 --dataset double_arrows
  ;;
*)
  echo "Invalid dataset"
  ;;
esac