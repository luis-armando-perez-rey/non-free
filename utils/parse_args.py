import argparse


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42, help="Set seed for training")

    # Training details
    parser.add_argument("--gpu", type=str, default="0", help="GPU to use")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--val-interval', type=int, default=10)
    parser.add_argument('--save-interval', default=1, type=int, help="Epoch to save model")

    # Dataset
    parser.add_argument('--dataset', default='multi-sprites', type=str, help="Dataset")
    parser.add_argument('--dataset_name', nargs="+", default=['4'], type=str, help="Dataset name")
    parser.add_argument('--batch-size', type=int, default=16, help="Batch size")

    parser.add_argument('--model-name', required=True, type=str, help="Name of model")
    parser.add_argument('--model', default='cnn', type=str, help="Model to use")
    parser.add_argument('--num', type=int, default=2, help="Number of modes or keypoints")
    parser.add_argument('--extra-dim', type=int, default=0, help="Orbit dimension")
    parser.add_argument('--latent-dim', type=int, default=2, help="Dimension N for SO(N)")
    parser.add_argument('--tau', type=float, default=1., help="Temperature of InfoNCE")
    parser.add_argument('--ndatapairs', type=int, default=-1, help="Number of pairs to use for training")

    parser.add_argument('--checkpoints-dir', default='checkpoints', type=str)

    # Optimization
    parser.add_argument('--optimizer', default="adam", help="Optimizer used")
    parser.add_argument('--lr-scheduler', action='store_true', default=False, help="Use a lr scheduler")
    parser.add_argument('--lr', default=1e-3, type=float)

    parser.add_argument('--data-dir', default='data', type=str)
    parser.add_argument('--equiv-loss', default="binary", type=str)
    parser.add_argument('--chamfer-reg', default=0.0, type=float)
    parser.add_argument('--weightequivariance', default=1.0, type=float)
    parser.add_argument('--use-comet', default=False, action='store_true')
    parser.add_argument('--neptune-user', default="", type=str, help="Neptune user if None, then Neptune is not used")

    # Model
    parser.add_argument('--identity-loss', default="infonce", type=str,
                        help="Loss to be used for identity representation")
    parser.add_argument('--enc-dist', default="gaussian-mixture", type=str, help="Distribution to use for encoder")
    parser.add_argument('--autoencoder', default="None", type=str, help="Autoencoder model to use")
    parser.add_argument('--decoder', default="None", type=str, help="Decoder model to use")
    parser.add_argument('--reconstruction-loss', default="bernoulli", type=str,
                        help="Loss to be used for reconstruction")
    parser.add_argument("--prior-dist", default="None", type=str, help="Prior distribution to use")
    parser.add_argument('--use-simplified', default=False, action='store_true',
                        help="Use simplified model that uses single encoder network")
    parser.add_argument('--no-norm-extra', default=False, action='store_true',
                        help="Don't normalize extra dimension")
    parser.add_argument('--attached-encoder', default=False, action='store_true',
                        help="Whether to attach encoder during training to backpropagate")
    parser.add_argument('--variablescale', default=False, action='store_true',
                        help="Use simplified model that uses single encoder network")
    # Plotting
    parser.add_argument("--plot", default=0, type=int, help="Number of epochs to ")

    # Neptune
    parser.add_argument("--neptunetags", nargs="+", default=[], type=str, help="Tags to use for neptune")

    return parser
