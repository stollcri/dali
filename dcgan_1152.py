# -*- coding: utf-8 -*-

import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
import tensorflow as tf
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from IPython import display
from progress.bar import FillingCirclesBar
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


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


class DeepConvolutionalGenerativeAdversarialNetwork(object):
    def __init__(self,
			source_dir=None,
			checkpoint_dir=None,
			target_dir=None,
			print_only=None,
			saved_model_dir=None,
			target_file=None,
            verbose=False
		):
        self.batch_size = 4
        self.epochs = 4000
        self.epochs_per_checkpoint = 32
        self.checkpoints_to_keep = 2

        self.image_height = 1152
        self.image_width = 1152
        self.image_depth = 3

        if source_dir is not None:
            self.source_images_path = pathlib.Path(source_dir)
            self.classes = [os.path.basename(x) for x in pathlib.Path(source_dir).iterdir() if x.is_dir()]
        if target_dir is not None:
            self.target_images_path = target_dir
        if target_file is not None:
            self.target_image_path = target_file
        self.saved_model_dir = saved_model_dir
        
        logger = logging.getLogger()
        if verbose:
            logger.setLevel(level=logging.DEBUG)
        else:
            logger.setLevel(level=logging.INFO)

        self.generator_input_shape = (32, 32, 3)
        
        self.generator = self.make_generator_model()
        self.discriminator = self.make_discriminator_model()

        self.seed = self.make_some_noise()

        if print_only:
            self.generator.build(input_shape=self.generator_input_shape)
            self.generator.summary()
            # # self.discriminator.build(input_shape=self.generator_input_shape)
            # self.discriminator.summary()
            exit()

        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

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
            max_to_keep=self.checkpoints_to_keep,
        )
        self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
        if self.checkpoint_manager.latest_checkpoint:
            logging.info(
                "Restored from {}".format(self.checkpoint_manager.latest_checkpoint)
            )
        else:
            logging.info("Initializing from scratch.")

    def make_some_noise(self):
        return tf.random.normal(
            [
                self.batch_size,
                self.generator_input_shape[0],
                self.generator_input_shape[1],
                self.generator_input_shape[2]
            ]
        )

    def make_generator_model(self):
        channel_axis = 1

        img_input = layers.Input(shape=self.generator_input_shape)
        x = layers.Flatten()(img_input)
        x = layers.Dense((3 * 3 * 2048), use_bias=False)(x)
        x = layers.BatchNormalization(axis=channel_axis)(x)
        x = layers.Activation('relu')(x)
        x = layers.Reshape((3, 3, 2048), input_shape=(3 * 3 * 2048,))(x)
                


        x = layers.Conv2DTranspose(2048, (3, 3), strides=(2, 2), data_format="channels_last", padding='same', use_bias=False, name='block14_sepconv2')(x)
        x = layers.BatchNormalization(axis=channel_axis, name='block14_sepconv2_bn')(x)
        x = layers.Activation('relu', name='block14_sepconv2_act')(x)
    
        x = layers.Conv2DTranspose(1024, (3, 3), strides=(2, 2), data_format="channels_last", padding='same', use_bias=False, name='block14_sepconv1')(x)
        x = layers.BatchNormalization(axis=channel_axis, name='block14_sepconv1_bn')(x)
        x = layers.Activation('relu', name='block14_sepconv1_act')(x)
        
        residual = layers.Conv2DTranspose(1024, (1, 1), strides=(1, 1), data_format="channels_last", padding='same', use_bias=False)(x)
        residual = layers.BatchNormalization(axis=channel_axis)(residual)
        
        x = layers.add([x, residual])
        x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block13_pool')(x)
        
        x = layers.Activation('relu', name='block13_sepconv2_act')(x)
        x = layers.Conv2DTranspose(728, (3, 3), strides=(2, 2), data_format="channels_last", padding='same', use_bias=False, name='block13_sepconv2')(x)
        x = layers.BatchNormalization(axis=channel_axis, name='block13_sepconv2_bn')(x)
        x = layers.Activation('relu', name='block13_sepconv1_act')(x)
        x = layers.Conv2DTranspose(512, (3, 3), strides=(2, 2), data_format="channels_last", padding='same', use_bias=False, name='block13_sepconv1')(x)
        x = layers.BatchNormalization(axis=channel_axis, name='block13_sepconv1_bn')(x)
    
        residual = layers.Conv2DTranspose(512, (1, 1), strides=(2, 2), data_format="channels_last", padding='same', use_bias=False)(x)
        residual = layers.BatchNormalization(axis=channel_axis)(residual)
        
        for i in range(4):
            residual = x
            prefix = 'block' + str(i + 5)
    
            x = layers.Activation('relu', name=prefix + '_sepconv1_act')(x)
            x = layers.Conv2DTranspose(512, (3, 3), data_format="channels_last", padding='same', use_bias=False, name=prefix + '_sepconv1')(x)
            x = layers.BatchNormalization(axis=channel_axis,name=prefix + '_sepconv1_bn')(x)
            x = layers.Activation('relu', name=prefix + '_sepconv2_act')(x)
            x = layers.Conv2DTranspose(512, (3, 3), data_format="channels_last", padding='same', use_bias=False, name=prefix + '_sepconv2')(x)
            x = layers.BatchNormalization(axis=channel_axis, name=prefix + '_sepconv2_bn')(x)
            x = layers.Activation('relu', name=prefix + '_sepconv3_act')(x)
            x = layers.Conv2DTranspose(512, (3, 3), data_format="channels_last", padding='same', use_bias=False, name=prefix + '_sepconv3')(x)
            x = layers.BatchNormalization(axis=channel_axis, name=prefix + '_sepconv3_bn')(x)
    
            x = layers.add([x, residual])
            
        x = layers.add([x, residual])
        x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block4_pool')(x)
            
        x = layers.Activation('relu', name='block4_sepconv1_act')(x)
        x = layers.Conv2DTranspose(512, (3, 3), strides=(2, 2), data_format="channels_last", padding='same', use_bias=False, name='block4_sepconv1')(x)
        x = layers.BatchNormalization(axis=channel_axis, name='block4_sepconv1_bn')(x)
        x = layers.Activation('relu', name='block4_sepconv2_act')(x)
        x = layers.Conv2DTranspose(512, (3, 3), strides=(2, 2), data_format="channels_last", padding='same', use_bias=False, name='block4_sepconv2')(x)
        x = layers.BatchNormalization(axis=channel_axis, name='block4_sepconv2_bn')(x)
            
        residual = layers.Conv2DTranspose(512, (1, 1), strides=(1, 1), data_format="channels_last", padding='same', use_bias=False)(x)
        residual = layers.BatchNormalization(axis=channel_axis)(residual)
        
        x = layers.add([x, residual])
        x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block3_pool')(x)
        
        x = layers.Activation('relu', name='block3_sepconv1_act')(x)
        x = layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), data_format="channels_last", padding='same', use_bias=False, name='block3_sepconv1')(x)
        x = layers.BatchNormalization(axis=channel_axis, name='block3_sepconv1_bn')(x)
        x = layers.Activation('relu', name='block3_sepconv2_act')(x)
        x = layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), data_format="channels_last", padding='same', use_bias=False, name='block3_sepconv2')(x)
        x = layers.BatchNormalization(axis=channel_axis, name='block3_sepconv2_bn')(x)
        
        residual = layers.Conv2DTranspose(256, (1, 1), strides=(1, 1), data_format="channels_last", padding='same', use_bias=False)(x)
        residual = layers.BatchNormalization(axis=channel_axis)(residual)
    
        x = layers.add([x, residual])
        x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block2_pool')(x)
        
        x = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), data_format="channels_last", padding='same', use_bias=False, name='block2_sepconv1')(x)
        x = layers.BatchNormalization(axis=channel_axis, name='block2_sepconv1_bn')(x)
        x = layers.Activation('relu', name='block2_sepconv2_act')(x)
        x = layers.Conv2DTranspose(128, (3, 3), data_format="channels_last", padding='same', use_bias=False, name='block2_sepconv2')(x)
        x = layers.BatchNormalization(axis=channel_axis, name='block2_sepconv2_bn')(x)
        
        x = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), data_format="channels_last", padding='same', use_bias=False, name='block1_conv2')(x)
        x = layers.BatchNormalization(axis=channel_axis, name='block1_conv2_bn')(x)
        x = layers.Activation('relu', name='block1_conv2_act')(x)
        x = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), data_format="channels_last", padding='same', use_bias=False, name='block1_conv1')(x)
        x = layers.BatchNormalization(axis=channel_axis, name='block1_conv1_bn')(x)
        x = layers.Activation('relu', name='block1_conv1_act')(x)



        x = layers.Conv2DTranspose(3, (3, 3), strides=(3, 3), data_format="channels_last", padding='same', use_bias=False)(x)
        x = layers.BatchNormalization(axis=channel_axis)(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2DTranspose(3, (3, 3), strides=(1, 1), activation="sigmoid", data_format="channels_last", padding="same", use_bias=False,)(x)
        x = layers.experimental.preprocessing.Rescaling(255)(x),
        return tf.keras.models.Model(img_input, x, name='xceptional')
    
    def make_discriminator_model(self):
        # return tf.keras.applications.MobileNetV2(
        #     input_shape=(self.image_height, self.image_width, self.image_depth),
        #     alpha=1.0,
        #     include_top=True,
        #     weights=None,
        #     input_tensor=None,
        #     pooling=None,
        #     classes=len(self.classes),
        #     classifier_activation="softmax",
        # )
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

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    def generate_and_save_images(self, model, input, file_name, print_multiple=False):
        # Notice `training` is set to False.
        # This is so all layers run in inference mode (batchnorm).
        predictions = model(input, training=False)[0]

        preds = predictions[0, :, :, :].numpy()
        logging.debug(f"Results range: {np.min(preds)} - {np.max(preds)}")

        if print_multiple:
            fig = plt.figure(figsize=(6, 6))
            for i in range(predictions.shape[0]):
                plt.subplot(2, 2, i + 1)
                plt.imshow(predictions[i, :, :, :].numpy().astype("uint8"))
                plt.axis("off")
            plt.savefig(file_name)
            plt.close()

        else:
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

    # Notice the use of `tf.function`
    # This annotation causes the function to be "compiled".
    @tf.function
    @tf.autograph.experimental.do_not_convert
    def train_step(self, images):
        noise = self.make_some_noise()

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)
            # self.generator.summary()

            real_output = self.discriminator(images[0], training=True)
            fake_output = self.discriminator(generated_images, training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(
            gen_loss, self.generator.trainable_variables
        )
        gradients_of_discriminator = disc_tape.gradient(
            disc_loss, self.discriminator.trainable_variables
        )

        self.generator_optimizer.apply_gradients(
            zip(gradients_of_generator, self.generator.trainable_variables)
        )
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables)
        )

        return gen_loss, disc_loss

    def train(self, dataset=None, epochs=None):
        if dataset is None:
            dataset = tf.keras.preprocessing.image_dataset_from_directory(
                self.source_images_path,
                image_size=(self.image_height, self.image_width),
                batch_size=self.batch_size,
            )
            # print(dataset)
            AUTOTUNE = tf.data.experimental.AUTOTUNE
            dataset = dataset.cache().shuffle(256).prefetch(buffer_size=AUTOTUNE)

        if epochs is None:
            epochs = self.epochs

        for epoch in range(epochs):
            start = time.time()

            bar = FillingCirclesBar(
                f"Epoch {epoch}/{epochs} Loss: gen {0.0:7.5f}, disc {0.0:7.5f}",
                max=len(dataset),
            )
            for image_batch in dataset:
                gen_loss, disc_loss = self.train_step(image_batch)
                bar.message = f"Epoch {epoch}/{epochs} Loss: gen {gen_loss.numpy():7.5f}, disc {disc_loss.numpy():7.5f}"
                bar.next()
            bar.finish()

            # Produce images for the GIF as we go
            display.clear_output(wait=True)
            image_file_name = "seed_{:04d}.png".format(epoch + 1)
            self.generate_and_save_images(
                self.generator,
                self.seed,
                os.path.join(self.target_images_path, image_file_name),
                True,
            )

            # Save the model every n epochs
            if (epoch + 1) % self.epochs_per_checkpoint == 0:
                self.checkpoint_manager.save()

            logging.debug(f"Epoch completed in {display_time(time.time() - start)}")

        # Generate after the final epoch
        display.clear_output(wait=True)
        image_file_name = "seed_{:04d}.png".format(epoch + 1)
        self.generate_and_save_images(
            self.generator,
            self.seed,
            os.path.join(self.target_images_path, image_file_name),
            True,
        )

    def draw(self, dataset=None, epochs=None):
        self.generate_and_save_images(
            self.generator,
            self.seed,
            self.target_image_path
        )
        
    def save(self):
        if self.saved_model_dir is not None:
            self.generator.save(self.saved_model_dir)
        else:
            logging.error("Source model directory not defined")
