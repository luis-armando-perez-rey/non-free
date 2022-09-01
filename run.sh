#!/bin/sh
python3 main.py --model-name 'TMP' --model 'resnet' --dataset 'rot-arrows'  --num 7 --extra-dim 0 --batch-size 256 --lr 1e-4 --epochs 100 --data-dir ./data
