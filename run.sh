#!/bin/bash
source /home/TUE/s161416/miniconda3/etc/profile.d/conda.sh
source activate gpytorch
identity_loss="infonce"
dataset_names=("1" "2" "3" "5" "7")
dataset_names=("1_c" "2_c" "3_c" "5_c" "7_c")
N_list=(2 3 4 6 8)
dataset="arrows"
extra_dim=2
experiment_id="von_mises"
epochs=100
optimizer="adamw"
model="resnet"
save_interval=1
lr=1e-4
batch_size=36
equiv_loss="cross-entropy"
seed=17
autoencoder="None"
reconstruction_loss="bernoulli"
enc_dist="von-mises-mixture"
prior_dist="von-mises-mixture"
decoder="resnet"

for num_dataset in "${!dataset_names[@]}"
do
  echo "Running experiment for dataset $dataset with $dataset_name pairs and repetition $repetition"
  echo $model_name
  N=${N_list[$num_dataset]}
  dataset_name=${dataset_names[$num_dataset]}
  model_name="D${dataset}-D${dataset_name// /_}-L$identity_loss-ED$extra_dim-N$N-M$model-A$model-ID$experiment_id-S$seed"
  python3 main.py --enc-dist ${enc_dist} --prior-dist ${prior_dist} --seed ${seed} --decoder ${decoder} --reconstruction-loss ${reconstruction_loss} --autoencoder ${autoencoder} --model ${model} --model-name ${model_name} --lr ${lr} --checkpoints-dir "./saved_models" --optimizer ${optimizer} --dataset ${dataset} --identity-loss ${identity_loss} --dataset_name ${dataset_name}  --num ${N} --extra-dim ${extra_dim} --batch-size ${batch_size} --epochs ${epochs} --data-dir ./data --save-interval ${save_interval} --plot 0 --equiv-loss ${equiv_loss}
  python3 visualize_latent_space.py --save-folder=${model_name} --dataset=${dataset} --dataset_name ${dataset_name}
  # In this experiments increase the number of arrows for each dataset
  N=$((N+1))
done
