#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import os
import sys

DALI_IMAGE_SIZE = os.environ.get("DALI_IMAGE_SIZE")
sys.path.append(os.path.abspath("."))

if DALI_IMAGE_SIZE == "1080":
    logging.info("Using dcgan_1080")
    from dcgan_1080 import DeepConvolutionalGenerativeAdversarialNetwork
elif DALI_IMAGE_SIZE == "1024":
    logging.info("Using dcgan_1024")
    from dcgan_1024 import DeepConvolutionalGenerativeAdversarialNetwork
elif DALI_IMAGE_SIZE == "360":
    logging.info("Using dcgan_360")
    from dcgan_360 import DeepConvolutionalGenerativeAdversarialNetwork
elif DALI_IMAGE_SIZE == "128":
    logging.info("Using dcgan_128")
    from dcgan_128 import DeepConvolutionalGenerativeAdversarialNetwork
elif DALI_IMAGE_SIZE == "90":
    logging.info("Using dcgan_90")
    from dcgan_90 import DeepConvolutionalGenerativeAdversarialNetwork
else:
    logging.info("Using dcgan_64")
    from dcgan_64 import DeepConvolutionalGenerativeAdversarialNetwork


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create images from a trained Deep Convolutional Neural Network")
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
        "-c", "--count",
        type=int,
        help="The number of images to generate",
        dest="count",
        default=1,
    )
    parser.add_argument("-v", "--verbose", help="verbose output", dest="verbose", action="store_true")
    args = parser.parse_args()

    if args.checkpoint_dir is None or args.target_file is None:
        parser.print_usage()
        exit()

    if args.verbose:
        logging.basicConfig(format="%(message)s", level=logging.DEBUG)
    else:
        logging.basicConfig(format="%(message)s", level=logging.INFO)

    dcnna = DeepConvolutionalGenerativeAdversarialNetwork(
        checkpoint_dir=args.checkpoint_dir, target_file=args.target_file, verbose=args.verbose
    )
    
    for n in range(args.count):
        file_name, file_extension = os.path.splitext(args.target_file)
        dcnna.draw(f"{file_name}-{n:02d}{file_extension}")
