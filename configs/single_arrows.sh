#!/bin/bash
arrow_numbers=(1 2 3 4 5)
ndatapairs_list=(2500 2500 2500 2500 2500 2500 2500 2500 2500 2500)
N_list=(1 2 3 4 5 6 7 8 9 10)
dataset="arrows"
extra_dim=0
epochs=50
weightequivariance=1
seed=$1
gpu=0

equiv_loss="chamfer"
batch_size=100
latent_dim=2
optimizer="adamw"
lr=1e-4
model="resnet"
autoencoder="ae"
enc_dist="gaussian-mixture"

# Fixed parameters
chamfer_reg=1
prior_dist="gaussian-mixture"
reconstruction_loss="bernoulli"
decoder="resnet"
save_interval=1
identity_loss="infonce"
experiment_id="results"

# Iterate over all arrow numbers 1,2,3,4,5
for arrow_number in "${arrow_numbers[@]}"
do
  dataset_name="${arrow_number}"
  # Iterate over all experiments with N = 1,2,3,4,5,6,7,8,9,10
  for num_experiment in "${!N_list[@]}"
  do
    echo "Running experiment for dataset $dataset with $dataset_name pairs and repetition $repetition"
    echo $model_name
    N=${N_list[$num_experiment]}
    ndatapairs=${ndatapairs_list[$num_experiment]}
    model_name="${dataset}/${experiment_id}/D${dataset}-D${dataset_name// /_}-L$identity_loss-ED$extra_dim-N$N-M$model-A$model-ID$experiment_id-S$seed"
    python3 main.py --neptune-user ${neptune_user} --gpu ${gpu} --ndatapairs ${ndatapairs} --chamfer-reg ${chamfer_reg} --weightequivariance ${weightequivariance} --enc-dist ${enc_dist} --latent-dim ${latent_dim} --prior-dist ${prior_dist} --seed ${seed} --decoder ${decoder} --reconstruction-loss ${reconstruction_loss} --autoencoder ${autoencoder} --model ${model} --model-name ${model_name} --lr ${lr} --checkpoints-dir "./saved_models" --optimizer ${optimizer} --dataset ${dataset} --identity-loss ${identity_loss} --dataset_name ${dataset_name}  --num ${N} --extra-dim ${extra_dim} --batch-size ${batch_size} --epochs ${epochs} --data-dir ./data --save-interval ${save_interval} --plot 0 --equiv-loss ${equiv_loss}
    python3 visualize_latent_space.py --save-folder=${model_name} --dataset=${dataset} --dataset_name ${dataset_name}
  done
done

