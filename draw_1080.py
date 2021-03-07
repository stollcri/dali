#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
# import pathlib
import tensorflow as tf
# import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# from IPython import display
# from progress.bar import FillingCirclesBar
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


class DeepConvolutionalNeuralNetworkArtist(object):
    def __init__(self, checkpoint_dir, target_file):
        self.image_height = 1080
        self.image_width = 1080
        self.image_depth = 3

        self.target_image_path = target_file
        
        self.generator = self.make_generator_model()
        self.discriminator = self.make_discriminator_model()
        
        self.seed = self.make_some_noise()

        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
        
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(
            generator_optimizer=self.generator_optimizer,
            discriminator_optimizer=self.discriminator_optimizer,
            generator=self.generator,
            discriminator=self.discriminator,
        )
        self.checkpoint_manager = tf.train.CheckpointManager(
            self.checkpoint,
            self.checkpoint_dir,
            max_to_keep=1,
        )
        self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
        if self.checkpoint_manager.latest_checkpoint:
            logging.info(
                "Restored from {}".format(self.checkpoint_manager.latest_checkpoint)
            )
        else:
            logging.error(
                "Checkpoint not found at {}".format(self.checkpoint_dir)
            )
            exit()

    def make_some_noise(self):
        return tf.random.normal(
            [1, self.image_height, self.image_width, self.image_depth]
        )

    def make_generator_model(self):
        model = tf.keras.Sequential(
            [
                layers.experimental.preprocessing.Resizing(
                    20,
                    20,
                    interpolation="bilinear",
                    input_shape=(self.image_height, self.image_width, self.image_depth),
                ),
				# 20 * 20 * 3 = 1,200
                layers.Flatten(),
                layers.Dense(45 * 45 * 36, use_bias=False),
				# 45 * 45 * 36 = 72,900
                layers.BatchNormalization(),
                layers.LeakyReLU(),
				layers.Reshape((45, 45, 36), input_shape=(45 * 45 * 36,)),
                layers.Conv2DTranspose(
                    34,
                    (3, 1),
                    strides=(2, 1),
                    activation="relu",
                    data_format="channels_last",
                    padding="same",
                    use_bias=False,
                ),
				# 90 * 45 * 34 = 137,700
                layers.Conv2DTranspose(
                    32,
                    (1, 3),
                    strides=(1, 2),
                    activation="relu",
                    data_format="channels_last",
                    padding="same",
                    use_bias=False,
                ),
				# 90 * 90 * 32 = 259,200
				layers.BatchNormalization(),
                layers.LeakyReLU(),
                layers.Conv2DTranspose(
                    25,
                    (3, 1),
                    strides=(3, 1),
                    activation="relu",
                    data_format="channels_last",
                    padding="same",
                    use_bias=False,
                ),
				# 270 * 90 * 25 = 607,500
                layers.Conv2DTranspose(
                    18,
                    (1, 3),
                    strides=(1, 3),
                    activation="relu",
                    data_format="channels_last",
                    padding="same",
                    use_bias=False,
                ),
				# 270 * 270 * 18 = 1,312,200
                layers.BatchNormalization(),
                layers.LeakyReLU(),
                layers.Conv2DTranspose(
                    12,
                    (3, 1),
                    strides=(2, 1),
                    activation="relu",
                    data_format="channels_last",
                    padding="same",
                    use_bias=False,
                ),
				# 540 * 270 * 12 = 1,749,600
                layers.Conv2DTranspose(
                    9,
                    (1, 3),
                    strides=(1, 2),
                    activation="relu",
                    data_format="channels_last",
                    padding="same",
                    use_bias=False,
                ),
				# 540 * 540 * 9 = 2,624,400
                layers.BatchNormalization(),
                layers.LeakyReLU(),
                layers.Conv2DTranspose(
                    3,
                    (3, 3),
                    strides=(2, 2),
                    activation="sigmoid",
                    data_format="channels_last",
                    padding="same",
                    use_bias=False,
                ),
				# 1080 * 1080 * 3 = 3,499,200
                layers.experimental.preprocessing.Rescaling(255),
            ]
        )
        return model

    def make_discriminator_model(self):
        model = tf.keras.Sequential(
            [
                layers.experimental.preprocessing.Rescaling(
                    1.0 / 255,
                    input_shape=(self.image_height, self.image_width, self.image_depth),
                ),
                # 1080 * 1080 * 3 = 3,499,200
                layers.Conv2D(
                    6,
                    (1, 3),
					strides=(1, 2),
                    activation="relu",
                    data_format="channels_last",
                    padding="same",
                ),
				# 1080 * 540 * 6 = 3,499,200
				layers.Conv2D(
                    9,
                    (3, 1),
					strides=(2, 1),
                    activation="relu",
                    data_format="channels_last",
                    padding="same",
                ),
				# 540 * 540 * 9 = 2,624,400
                layers.MaxPooling2D(),
				# 270 * 270 * 9 = 656,100
                layers.Dropout(0.2),
                layers.Conv2D(
                    12,
                    (1, 3),
					strides=(1, 3),
                    activation="relu",
                    data_format="channels_last",
                    padding="same",
                ),
				# 270 * 90 * 12 = 291,600
				layers.Conv2D(
                    18,
                    (3, 1),
					strides=(3, 1),
                    activation="relu",
                    data_format="channels_last",
                    padding="same",
                ),
				# 90 * 90 * 18 = 145,800
                layers.MaxPooling2D(),
				# 45 * 45 * 18 = 36,450
                layers.Dropout(0.2),
				layers.Conv2D(
                    36,
                    (3, 3),
					strides=(3, 3),
                    activation="relu",
                    data_format="channels_last",
                    padding="same",
                ),				
				# 15 * 15 * 36 = 8100
                layers.Dropout(0.5),
                layers.Flatten(),
                layers.Dense(128, activation="relu"),
                layers.Dense(1),
            ]
        )
        return model

    def generate_and_save_images(self, model, input, file_name):
        # Notice `training` is set to False.
        # This is so all layers run in inference mode (batchnorm).
        predictions = model(input, training=False)

        preds = predictions[0, :, :, :].numpy()
        logging.debug(f"Results range: {np.min(preds)} - {np.max(preds)}")

        sizes = np.shape(predictions[0, :, :, :].numpy().astype("uint8"))
        fig = plt.figure(figsize=(1, 1))
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(
            predictions[0, :, :, :].numpy().astype("uint8"),
            cmap=plt.get_cmap("bone"),
        )
        plt.savefig(file_name, dpi=sizes[0])
        plt.close()

    def draw(self, dataset=None, epochs=None):
        self.generate_and_save_images(
            self.generator,
            self.seed,
            self.target_image_path
        )


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

    dcnna = DeepConvolutionalNeuralNetworkArtist(
        args.checkpoint_dir, args.target_file
    )
    dcnna.draw()
