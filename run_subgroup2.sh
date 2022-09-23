#!/bin/bash
source /home/TUE/s161416/miniconda3/etc/profile.d/conda.sh
source activate gpytorch
identity_loss="infonce"
dataset_name="1_2"
dataset_names=("1_2" "2_2" "3_3")
dataset_names=("1_2 2_1")
dataset="sinusoidal_translation"
N_list=(2 4 6)
extra_dim=0
experiment_id="adamw"
latent_dim=4
epochs=100
optimizer="adamw"
model="resnet"
save_interval=1
lr=1e-4
batch_size=128
equiv_loss="cross-entropy"
seed=4
autoencoder="None"
reconstruction_loss="bernoulli"
enc_dist="gaussian-mixture"
decoder="resnet"
prior_dist="gaussian-mixture"
for num_dataset in "${!dataset_names[@]}"
do
  echo "Running experiment for dataset $dataset with $dataset_name pairs and repetition $repetition"
  echo $model_name
  N=${N_list[$num_dataset]}
  dataset_name=${dataset_names[$num_dataset]}
  model_name="D${dataset}-D${dataset_name// /_}-L$identity_loss-ED$extra_dim-N$N-M$model-A$model-ID$experiment_id-S$seed"
  python3 main.py --enc-dist ${enc_dist} --latent-dim ${latent_dim} --prior-dist ${prior_dist} --seed ${seed} --decoder ${decoder} --reconstruction-loss ${reconstruction_loss} --autoencoder ${autoencoder} --model ${model} --model-name ${model_name} --lr ${lr} --checkpoints-dir "./saved_models" --optimizer ${optimizer} --dataset ${dataset} --identity-loss ${identity_loss} --dataset_name ${dataset_name}  --num ${N} --extra-dim ${extra_dim} --batch-size ${batch_size} --epochs ${epochs} --data-dir ./data --save-interval ${save_interval} --plot 0 --equiv-loss ${equiv_loss}
  python3 visualize_latent_space.py --save-folder=${model_name} --dataset=${dataset} --dataset_name ${dataset_name}
  # In this experiments increase the number of arrows for each dataset
  N=$((N+1))
done