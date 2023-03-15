#!/bin/bash
source /home/TUE/s161416/miniconda3/etc/profile.d/conda.sh
source activate gpytorch


# Non-fixed parameters
dataset_names=("1_c 2_c 3_c 4_c 5_c" "1_c 2_c 3_c 4_c 5_c" "1_c 2_c 3_c 4_c 5_c" "1_c 2_c 3_c 4_c 5_c" "1_c 2_c 3_c 4_c 5_c" "1_c 2_c 3_c 4_c 5_c" "1_c 2_c 3_c 4_c 5_c" "1_c 2_c 3_c 4_c 5_c" "1_c 2_c 3_c 4_c 5_c" "1_c 2_c 3_c 4_c 5_c" "1_c 2_c 3_c 4_c 5_c" "1_c 2_c 3_c 4_c 5_c" "1_c 2_c 3_c 4_c 5_c" "1_c 2_c 3_c 4_c 5_c" "1_c 2_c 3_c 4_c 5_c" "1_c 2_c 3_c 4_c 5_c" "1_c 2_c 3_c 4_c 5_c" "1_c 2_c 3_c 4_c 5_c" "1_c 2_c 3_c 4_c 5_c" "1_c 2_c 3_c 4_c 5_c" "1_c 2_c 3_c 4_c 5_c" "1_c 2_c 3_c 4_c 5_c" "1_c 2_c 3_c 4_c 5_c" "1_c 2_c 3_c 4_c 5_c" "1_c 2_c 3_c 4_c 5_c" "1_c 2_c 3_c 4_c 5_c" "1_c 2_c 3_c 4_c 5_c" "1_c 2_c 3_c 4_c 5_c" "1_c 2_c 3_c 4_c 5_c" "1_c 2_c 3_c 4_c 5_c" "1_c 2_c 3_c 4_c 5_c" "1_c 2_c 3_c 4_c 5_c" "1_c 2_c 3_c 4_c 5_c" "1_c 2_c 3_c 4_c 5_c" "1_c 2_c 3_c 4_c 5_c" "1_c 2_c 3_c 4_c 5_c" "1_c 2_c 3_c 4_c 5_c" "1_c 2_c 3_c 4_c 5_c" "1_c 2_c 3_c 4_c 5_c" "1_c 2_c 3_c 4_c 5_c" "1_c 2_c 3_c 4_c 5_c" "1_c 2_c 3_c 4_c 5_c" "1_c 2_c 3_c 4_c 5_c" "1_c 2_c 3_c 4_c 5_c" "1_c 2_c 3_c 4_c 5_c" "1_c 2_c 3_c 4_c 5_c" "1_c 2_c 3_c 4_c 5_c" "1_c 2_c 3_c 4_c 5_c" "1_c 2_c 3_c 4_c 5_c" "1_c 2_c 3_c 4_c 5_c" "1_c 2_c 3_c 4_c 5_c" "1_c 2_c 3_c 4_c 5_c" "1_c 2_c 3_c 4_c 5_c" "1_c 2_c 3_c 4_c 5_c" "1_c 2_c 3_c 4_c 5_c" "1_c 2_c 3_c 4_c 5_c" "1_c 2_c 3_c 4_c 5_c" "1_c 2_c 3_c 4_c 5_c" "1_c 2_c 3_c 4_c 5_c" "1_c 2_c 3_c 4_c 5_c")
#dataset_names=("stool_0 chair_0 airplane_0" )
dataset_names=("1_c 2_c 3_c 4_c 5_c" "1_c 2_c 3_c 4_c 5_c")

ndatapairs_list=(-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1)
#ndatapairs_list=(2500)
#ndatapairs_list=(5000 5000 5000 5000 5000)
N_list=(5 5 5 5 5)
#N_list=(10 10 10 10 10)
dataset="arrows"
extra_dim=10
epochs=200
weightequivariance=1
seeds=(17 17 17 17 17 17 17 17 17 17 42 42 42 42 42 42 42 42 42 42 28 28 28 28 28 28 28 28 28 28 19 19 19 19 19 19 19 19 19 19 58 58 58 58 58 58 58 58 58 58)
seeds=(17 42)
gpu=0



# Semi-fixed parameters
equiv_loss="cross-entropy"
batch_size=100
latent_dim=2
optimizer="adamw"
lr=1e-4
model="resnet"
autoencoder="vae"
enc_dist="von-mises-mixture"


# Fixed parameters
chamfer_reg=1.0
prior_dist="von-mises-mixture"
reconstruction_loss="bernoulli"
decoder="resnet"
save_interval=1
identity_loss="infonce"
neptune_user="laprhanabi"
experiment_id="colorvae"

for num_dataset in "${!dataset_names[@]}"
do
  echo "Running experiment for dataset $dataset with $dataset_name pairs and repetition $repetition"
  echo $model_name
  N=${N_list[$num_dataset]}
  seed=${seeds[$num_dataset]}
  dataset_name=${dataset_names[$num_dataset]}
  ndatapairs=${ndatapairs_list[$num_dataset]}
  model_name="${dataset}/${experiment_id}/D${dataset}-D${dataset_name// /_}-L$identity_loss-ED$extra_dim-N$N-M$model-A$model-ID$experiment_id-S$seed"
  python3 main.py --neptune-user ${neptune_user} --gpu ${gpu} --ndatapairs ${ndatapairs} --chamfer-reg ${chamfer_reg} --weightequivariance ${weightequivariance} --enc-dist ${enc_dist} --latent-dim ${latent_dim} --prior-dist ${prior_dist} --seed ${seed} --decoder ${decoder} --reconstruction-loss ${reconstruction_loss} --autoencoder ${autoencoder} --model ${model} --model-name ${model_name} --lr ${lr} --checkpoints-dir "./saved_models" --optimizer ${optimizer} --dataset ${dataset} --identity-loss ${identity_loss} --dataset_name ${dataset_name}  --num ${N} --extra-dim ${extra_dim} --batch-size ${batch_size} --epochs ${epochs} --data-dir ./data --save-interval ${save_interval} --plot 0 --equiv-loss ${equiv_loss}
  python3 visualize_latent_space.py --save-folder=${model_name} --dataset=${dataset} --dataset_name ${dataset_name}
  # In this experiments increase the number of arrows for each dataset
done

