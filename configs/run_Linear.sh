#!/bin/bash
# source /home/TUE/s161416/miniconda3/etc/profile.d/conda.sh
# source activate gpytorch
dataset_name="airplane_0 stool_0 chair_0 lamp_0 bathtub_0"
dataset="modelnet"
latent_dim=2
experiment_id="test"
epochs=50
optimizer="adamw"
save_interval=1
lr=1e-4
batch_size=100
neptune_user="laprhanabi"
seeds=(17 42 58 28 19)
ndatapairs=-1
gpu=0



model_name="${dataset}/${experiment_id}/D${dataset}-D${dataset_name// /_}-$experiment_id"
python3 main_Linear.py --model-name ${model_name} --lr ${lr} --checkpoints-dir "./saved_models/linear" --optimizer ${optimizer} --dataset ${dataset} --dataset_name ${dataset_name}  --batch-size ${batch_size} --epochs ${epochs} --data-dir ./data --save-interval ${save_interval} --plot 0 --latent-dim ${latent_dim} --neptune-user ${neptune_user}
# python3 visualize_latent_space.py --save-folder=${model_name} --dataset ${dataset} --dataset_name ${dataset_name}
