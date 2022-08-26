#!/bin/sh
identity_loss="infonce"
dataset_name="1_m"
N=1
extra_dim=0
experiment_id="1"
epochs=200
model_name="D$dataset_name-L$identity_loss-ED$extra_dim-N$N-ID$experiment_id"
python3 main.py --model-name ${model_name} --checkpoints-dir "./saved_models" --model 'cnn' --dataset 'rot-arrows' --identity-loss ${identity_loss} --dataset_name ${dataset_name}  --num ${N} --extra-dim ${extra_dim} --batch-size 128 --epochs ${epochs} --data-dir ./data
python3 visualize_latent_space.py --save-folder=${model_name} --dataset=rot-arrows --dataset_name ${dataset_name}