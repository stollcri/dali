#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import os
import sys

sys.path.append(os.path.abspath("."))

from dcgan_1080 import DeepConvolutionalGenerativeAdversarialNetwork


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create images from a trained Deep Convolutional Neural Network"
    )
    parser.add_argument(
        metavar="CHECKPOINT_DIRECTORY",
        help="Directory to read checkpoint",
        dest="checkpoint_dir",
        default=None,
    )
    parser.add_argument(
		metavar="FILENAME",
        help="Name of resulting image",
        dest="target_file",
        default=None,
    )
    parser.add_argument(
        "-v", "--verbose", help="verbose output", dest="verbose", action="store_true"
    )
    args = parser.parse_args()

    if (args.checkpoint_dir is None or args.target_file is None):
        parser.print_usage()
        exit()

    if args.verbose:
        logging.basicConfig(format="%(message)s", level=logging.DEBUG)
    else:
        logging.basicConfig(format="%(message)s", level=logging.INFO)

    dcnna = DeepConvolutionalGenerativeAdversarialNetwork(
        checkpoint_dir=args.checkpoint_dir, target_file=args.target_file
    )
    dcnna.draw()
