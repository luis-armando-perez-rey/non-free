import argparse
import torch
import torch.nn.functional as F


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42, help="Set seed for training")

    # Training details
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--val-interval', type=int, default=10)
    parser.add_argument('--save-interval', default=1, type=int, help="Epoch to save model")

    # Dataset
    parser.add_argument('--dataset', default='multi-sprites', type=str, help="Dataset")
    parser.add_argument('--dataset_name', default='4', type=str, help="Dataset name")
    parser.add_argument('--batch-size', type=int, default=16, help="Batch size")

    parser.add_argument('--model-name', required=True, type=str, help="Name of model")
    parser.add_argument('--model', default='cnn', type=str, help="Model to use")
    parser.add_argument('--num', type=int, default=2, help="Number of modes or keypoints")
    parser.add_argument('--extra-dim', type=int, default=0, help="Orbit dimension")
    parser.add_argument('--tau', type=float, default=1., help="Temperature of InfoNCE")


    parser.add_argument('--checkpoints-dir', default='checkpoints', type=str)

    # Optimization
    parser.add_argument('--lr-scheduler', action='store_true', default=False, help="Use a lr scheduler")
    parser.add_argument('--lr', default=1e-3, type=float)

    parser.add_argument('--data-dir', default='data', type=str)


    parser.add_argument('--use-comet', default=False, action='store_true')


    return parser
