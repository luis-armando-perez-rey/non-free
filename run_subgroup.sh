#!/bin/bash
identity_loss="infonce"
dataset_name="1_1"
dataset="sinusoidal_translation"
N=1
extra_dim=0
experiment_id=""
latent_dim=4
epochs=100
optimizer="adamw"
model="resnet"
save_interval=1
lr=1e-4
batch_size=128
equiv_loss="chamfer"
seed=4
autoencoder="None"
reconstruction_loss="bernoulli"
decoder="resnet"
model_name="D${dataset}-D${dataset_name// /_}-L$identity_loss-ED$extra_dim-N$N-M$model-A$model-ID$experiment_id"
python3 main.py --num ${N} --latent-dim ${latent_dim} --seed ${seed} --decoder ${decoder} --reconstruction-loss ${reconstruction_loss} --autoencoder ${autoencoder} --model ${model} --model-name ${model_name} --lr ${lr} --checkpoints-dir "./saved_models" --optimizer ${optimizer} --dataset ${dataset} --identity-loss ${identity_loss} --dataset_name ${dataset_name} --extra-dim ${extra_dim} --batch-size ${batch_size} --epochs ${epochs} --data-dir ./data --save-interval ${save_interval} --plot 0 --equiv-loss ${equiv_loss}
python3 visualize_latent_space.py --save-folder=${model_name} --dataset=${dataset} --dataset_name ${dataset_name}
