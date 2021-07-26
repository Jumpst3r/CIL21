import argparse
import os.path as op


def parse_args_function():
    parser = argparse.ArgumentParser()

    parser.add_argument("--lr", type=float, default=1e-4, help='Learning rate')

    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help='seed')

    parser.add_argument("--nb_blocks",
                        type=int,
                        default=4,
                        help='Number of stacks')

    parser.add_argument("--unet_mode",
                        type=str,
                        default='classic-backbone',
                        choices=['classic', 'classic-backbone'],
                        help="Architecture")

    parser.add_argument("--stacking_mode",
                        type=str,
                        default='hourglass',
                        choices=['simple', 'hourglass'],
                        help="Type of stacking")

    parser.add_argument("--loss_mode",
                        type=str,
                        default='avg',
                        choices=['last', 'avg', 'sum'],
                        help="Type of loss")

    parser.add_argument("--max_epochs",
                        type=int,
                        default=150,
                        help="Maximum number of training epochs")

    parser.add_argument("--batch_size", type=int, default=5, help="Batch size")

    parser.add_argument("--res",
                        type=int,
                        default=128,
                        help="Training resolution")

    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default='',
        help=
        "Directory to store trained model. If '' is passed, no model is saved")

    args = parser.parse_args()
    root_dir = op.join('.')
    args.root_dir = root_dir
    return args


args = parse_args_function()
