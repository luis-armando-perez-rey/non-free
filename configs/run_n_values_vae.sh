#!/bin/bash
source /home/TUE/s161416/miniconda3/etc/profile.d/conda.sh
source activate gpytorch


# Non-fixed parameters
dataset_names=("3 4 5")
#dataset_names=("stool_0 chair_0 airplane_0" )
ndatapairs_list=(-1 -1 -1)
N_list=(5)
dataset="arrows"
extra_dim=3
epochs=100
weightequivariance=1
seeds=(17)
gpu=1




# Semi-fixed parameters
equiv_loss="cross-entropy"
batch_size=100
latent_dim=2
optimizer="adamw"
lr=1e-4
model="resnet"
autoencoder="vae"
enc_dist="gaussian-mixture"


# Fixed parameters
chamfer_reg=1.0
prior_dist="gaussian-mixture"
reconstruction_loss="bernoulli"
decoder="resnet"
save_interval=1
identity_loss="infonce"
neptune_user="laprhanabi"
experiment_id="vae"

for num_dataset in "${!dataset_names[@]}"
do
  echo "Running experiment for dataset $dataset with $dataset_name pairs and repetition $repetition"
  echo $model_name
  N=${N_list[$num_dataset]}
  seed=${seeds[$num_dataset]}
  dataset_name=${dataset_names[$num_dataset]}
  ndatapairs=${ndatapairs_list[$num_dataset]}
  model_name="${dataset}/${experiment_id}/D${dataset}-D${dataset_name// /_}-L$identity_loss-ED$extra_dim-N$N-M$model-A$model-ID$experiment_id-S$seed"
  python3 main.py --no-norm-extra --use-simplified --attached-encoder --neptunetags ${experiment_id} --neptune-user ${neptune_user} --gpu ${gpu} --ndatapairs ${ndatapairs} --chamfer-reg ${chamfer_reg} --weightequivariance ${weightequivariance} --enc-dist ${enc_dist} --latent-dim ${latent_dim} --prior-dist ${prior_dist} --seed ${seed} --decoder ${decoder} --reconstruction-loss ${reconstruction_loss} --autoencoder ${autoencoder} --model ${model} --model-name ${model_name} --lr ${lr} --checkpoints-dir "./saved_models" --optimizer ${optimizer} --dataset ${dataset} --identity-loss ${identity_loss} --dataset_name ${dataset_name}  --num ${N} --extra-dim ${extra_dim} --batch-size ${batch_size} --epochs ${epochs} --data-dir ./data --save-interval ${save_interval} --plot 0 --equiv-loss ${equiv_loss}
  python3 visualize_latent_space.py --save-folder=${model_name} --dataset=${dataset} --dataset_name ${dataset_name}
  # In this experiments increase the number of arrows for each dataset
done

