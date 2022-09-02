#!/bin/bash
identity_loss="infonce"
dataset_name="4 5 6 7"
dataset="rot-arrows"
N=7
extra_dim=0
experiment_id=""
epochs=100
optimizer="adamw"
model="resnet"
save_interval=1
lr=1e-4
batch_size=16
equiv_loss="binary"
model_name="D${dataset}-D${dataset_name// /_}-L$identity_loss-ED$extra_dim-N$N-M$model-A$model-ID$experiment_id"
python3 main.py --model ${model} --model-name ${model_name} --lr ${lr} --checkpoints-dir "./saved_models" --optimizer ${optimizer} --dataset ${dataset} --identity-loss ${identity_loss} --dataset_name ${dataset_name}  --num ${N} --extra-dim ${extra_dim} --batch-size ${batch_size} --epochs ${epochs} --data-dir ./data --save-interval ${save_interval} --plot 0 --equiv-loss ${equiv_loss}
python3 visualize_latent_space.py --save-folder=${model_name} --dataset=rot-arrows --dataset_name ${dataset_name}
