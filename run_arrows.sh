#!/bin/bash
source /home/TUE/s161416/miniconda3/etc/profile.d/conda.sh
source activate gpytorch
identity_loss="infonce"
dataset_name="4 5 6"
dataset="arrows"
N=50
extra_dim=0
experiment_id="test"
epochs=1
optimizer="adamw"
model="resnet"
save_interval=1
lr=1e-4
batch_size=128
equiv_loss="chamfer"
enc_dist="None"
latent_dim=2
model_name="${dataset}/${experiment_id}/D${dataset}-D${dataset_name// /_}-L$identity_loss-ED$extra_dim-N$N-M$model-A$model-ID$experiment_id"
#python3 main.py --enc-dist ${enc_dist} --model ${model} --model-name ${model_name} --lr ${lr} --checkpoints-dir "./saved_models" --optimizer ${optimizer} --dataset ${dataset} --identity-loss ${identity_loss} --dataset_name ${dataset_name}  --num ${N} --extra-dim ${extra_dim} --batch-size ${batch_size} --epochs ${epochs} --data-dir ./data --save-interval ${save_interval} --plot 0 --equiv-loss ${equiv_loss} --latent-dim ${latent_dim}
python3 visualize_latent_space.py --save-folder=${model_name} --dataset ${dataset} --dataset_name ${dataset_name}
