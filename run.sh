#!/bin/sh

#python3 main.py --model-name 'TMP' --checkpoints-dir "./saved_models" --model 'cnn' --dataset 'rot-arrows' --dataset_name "4"  --num 5 --extra-dim 0 --batch-size 128 --epochs 100 --data-dir ./data
python3 main.py --model-name 'TMP3-2' --checkpoints-dir "./saved_models" --model 'cnn' --dataset 'rot-arrows' --dataset_name "3"  --num 4 --extra-dim 0 --batch-size 128 --epochs 100 --data-dir ./data
