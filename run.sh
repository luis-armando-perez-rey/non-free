#!/bin/sh

python3 main.py --model-name 'TMP' --model 'resnet' --dataset 'rot-arrows'  --num 7 --extra-dim 0 --batch-size 16 --epochs 100 --data-dir ./data
