#!/bin/bash
# source /home/TUE/s161416/miniconda3/etc/profile.d/conda.sh
# source activate gpytorch
dataset_name="1 2 3 4 5"
dataset="arrows"
reconstruction_loss="bernoulli"
experiment_id="enr"
epochs=100
optimizer="adamw"
save_interval=1
lr=1e-4
batch_size=20
neptune_user="laprhanabi"
seeds=(17 42 58 28 19)
ndatapairs=-1
gpu=3
latent_dim=4
seeds=(17 42 58 28 19)

for seed in "${!seeds[@]}"
do
  model_name="${dataset}/${experiment_id}/D${dataset}-D${dataset_name// /_}-LD$latent_dim-ID$experiment_id-S$seed"
#  python3 main_ENR.py --neptunetags ${experiment_id} --latent-dim ${latent_dim} --neptune-user ${neptune_user} --ndatapairs ${ndatapairs} --model-name ${model_name} --lr ${lr} --checkpoints-dir "./saved_models" --gpu ${gpu} --optimizer ${optimizer} --reconstruction-loss ${reconstruction_loss} --dataset ${dataset} --dataset_name ${dataset_name}  --batch-size ${batch_size} --epochs ${epochs} --data-dir ./data --save-interval ${save_interval} --plot 0
  python3 visualize_enr.py --save-folder=${model_name} --dataset ${dataset} --dataset_name ${dataset_name}
done