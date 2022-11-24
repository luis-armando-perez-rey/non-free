#!/bin/bash
# source /home/TUE/s161416/miniconda3/etc/profile.d/conda.sh
# source activate gpytorch
dataset_name="tetrahedron"
dataset="symmetric_solids"
reconstruction_loss="bernoulli"
experiment_id="test"
epochs=50
optimizer="adamw"
save_interval=1
lr=1e-4
batch_size=8
model_name="${dataset}/${experiment_id}/D${dataset}-D${dataset_name// /_}-$experiment_id"
python3 main_ENR.py --model-name ${model_name} --lr ${lr} --checkpoints-dir "./saved_models_ENR" --optimizer ${optimizer} --reconstruction-loss ${reconstruction_loss} --dataset ${dataset} --dataset_name ${dataset_name}  --batch-size ${batch_size} --epochs ${epochs} --data-dir ./data --save-interval ${save_interval} --plot 0
# python3 visualize_latent_space.py --save-folder=${model_name} --dataset ${dataset} --dataset_name ${dataset_name}
