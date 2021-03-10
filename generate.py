#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import os
import sys

DALI_IMAGE_SIZE = os.environ.get('DALI_IMAGE_SIZE')
sys.path.append(os.path.abspath("."))

if DALI_IMAGE_SIZE == 1080:
    from dcgan_1080 import DeepConvolutionalGenerativeAdversarialNetwork
elif DALI_IMAGE_SIZE == 360:
    from dcgan_360 import DeepConvolutionalGenerativeAdversarialNetwork
else::
    from dcgan_90 import DeepConvolutionalGenerativeAdversarialNetwork

def display_time(seconds, granularity=3):
    intervals = (
        ("weeks", 604800),  # 60 * 60 * 24 * 7
        ("days", 86400),  # 60 * 60 * 24
        ("hours", 3600),  # 60 * 60
        ("minutes", 60),
        ("seconds", 1),
    )
    result = []
    for name, count in intervals:
        value = seconds // count
        if value:
            seconds -= value * count
            if value == 1:
                name = name.rstrip("s")
            result.append("{} {}".format(value, name))
    return ", ".join(result[:granularity])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate images using Deep Convolutional Generative Adversarial Network"
    )
    parser.add_argument(
        "-s",
        "--source-dir",
        help="Directory of class directories",
        dest="source_dir",
        default=None,
    )
    parser.add_argument(
        "-c",
        "--checkpoint-dir",
        help="Directory to store checkpoints",
        dest="checkpoint_dir",
        default=None,
    )
    parser.add_argument(
        "-t",
        "--target-dir",
        help="Directory of resulting images",
        dest="target_dir",
        default=None,
    )
    parser.add_argument(
        "-p", "--print", help="print model structure and exit", dest="print_only", action="store_true"
    )
    parser.add_argument(
        "-v", "--verbose", help="verbose output", dest="verbose", action="store_true"
    )
    args = parser.parse_args()

    if (
        args.checkpoint_dir is None
        or args.source_dir is None
        or args.target_dir is None
    ):
        parser.print_usage()
        exit()

    if args.verbose:
        logging.basicConfig(format="%(message)s", level=logging.DEBUG)
    else:
        logging.basicConfig(format="%(message)s", level=logging.INFO)

    dcgan = DeepConvolutionalGenerativeAdversarialNetwork(
        source_dir=args.source_dir,
        checkpoint_dir=args.checkpoint_dir,
        target_dir=args.target_dir,
        print_only=args.print_only
    )
    dcgan.train()
