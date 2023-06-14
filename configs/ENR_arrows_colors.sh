#!/bin/bash
dataset_name="1_c 2_c 3_c 4_c 5_c"
dataset="arrows"
reconstruction_loss="bernoulli"
experiment_id="enr"
epochs=100
optimizer="adamw"
save_interval=1
lr=1e-4
batch_size=20
ndatapairs=-1
gpu=0
latent_dim=8
# Save the first argument to seed variable
seed=$1

model_name="${dataset}/${experiment_id}/D${dataset}-D${dataset_name// /_}-LD$latent_dim-ID$experiment_id-S$seed"
python3 main_ENR.py --neptunetags ${experiment_id} --latent-dim ${latent_dim} --neptune-user ${neptune_user} --ndatapairs ${ndatapairs} --model-name ${model_name} --lr ${lr} --checkpoints-dir "./saved_models" --gpu ${gpu} --optimizer ${optimizer} --reconstruction-loss ${reconstruction_loss} --dataset ${dataset} --dataset_name ${dataset_name}  --batch-size ${batch_size} --epochs ${epochs} --data-dir ./data --save-interval ${save_interval} --plot 0
python3 visualize_enr.py --save-folder=${model_name} --dataset ${dataset} --dataset_name ${dataset_name}
