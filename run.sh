#!/bin/sh

python3 main.py --model-name 'TMP' --model 'cnn' --dataset 'rot-arrows'  --num 5 --extra-dim 0 --batch-size 128 --epochs 100 --data-dir ./data
