#!/bin/sh

python3 main.py --model-name 'TMP' --model 'resnet' --dataset 'rot-arrows'  --num 7 --extra-dim 2 --batch-size 8 --epochs 100 --data-dir ./data
