import argparse


def create_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--max_epochs", default=100, required=False, help="Epochs to train", type=int
    )
    parser.add_argument(
        "--learning_rate",
        default=1e-3,
        required=False,
        help="VAE learning rate",
        type=float,
    )
    parser.add_argument(
        "--beta", default=1, required=False, help="beta label loss weight", type=float
    )
    parser.add_argument("--use_decoder", action="store_true")

    parser.add_argument(
        "--batch_size",
        default=4,
        required=False,
        help="bacth size for train",
        type=int,
    )

    parser.add_argument(
        "--act",
        default="PRELU",
        required=False,
        help="Chose activation function PRELU or RELU",
        type=str,
    )
    parser.add_argument(
        "--conv_k", default=5, required=False, help="size of conv kernel", type=int
    )
    parser.add_argument(
        "--seed", default=None, required=False, help="Random seed for torch", type=int
    )
    parser.add_argument("-n", "--narval", action="store_true")
    return parser.parse_args()
