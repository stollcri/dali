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

def get_model_memory_usage(batch_size, model):
    shapes_mem_count = 0
    internal_model_mem_count = 0
    
    for l in model.layers:
        layer_type = l.__class__.__name__
        if layer_type == 'Model':
            internal_model_mem_count += get_model_memory_usage(batch_size, l)
        single_layer_mem = 1
        out_shape = l.output_shape
        if type(out_shape) is list:
            out_shape = out_shape[0]
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([keras.backend.count_params(p) for p in model.trainable_weights])
    non_trainable_count = np.sum([keras.backend.count_params(p) for p in model.non_trainable_weights])

    number_size = 4.0
    if keras.backend.floatx() == 'float16':
        number_size = 2.0
    if keras.backend.floatx() == 'float64':
        number_size = 8.0

    total_memory = number_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3) + internal_model_mem_count
    return gbytes

# TODO: Create and abstract base class
class DeepConvolutionalGenerativeAdversarialNetwork(object):
    def __init__(
        self,
        source_dir=None,
        checkpoint_dir=None,
        target_dir=None,
        print_only=None,
        saved_model_dir=None,
        target_file=None,
        verbose=False,
    ):
        self.batch_size = 9
        self.epochs = 4000
        self.epochs_per_checkpoint = 32
        self.checkpoints_to_keep = 2

        self.image_height = 360
        self.image_width = 360
        self.image_depth = 3

        if source_dir is not None:
            self.source_images_path = pathlib.Path(source_dir)
        if target_dir is not None:
            self.target_images_path = target_dir
        if target_file is not None:
            self.target_image_path = target_file
        self.saved_model_dir = saved_model_dir

        logger = logging.getLogger()
        console_formatter = logging.Formatter("%(message)s")
        if len(logger.handlers) > 0:
            console_handler = logger.handlers[0]
        else:
            console_handler = logging.StreamHandler()
            logger.addHandler(console_handler)
        console_handler.setFormatter(console_formatter)

        if verbose:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

        self.generator = self.make_generator_model()
        self.discriminator = self.make_discriminator_model()

        self.seed = self.make_some_noise()

        if print_only:
            # self.generator.build(input_shape=self.seed.shape)
            self.generator.summary()
            print()
            
            # self.discriminator.build(input_shape=self.seed.shape)
            self.discriminator.summary()
            print()
            
            generator_size = get_model_memory_usage(self.batch_size, self.generator)
            discriminator_size = get_model_memory_usage(self.batch_size, self.discriminator)
            print(f"Generator Model Size:        {generator_size} GB")
            print(f"Discriminator Model Size:    {discriminator_size} GB")
            # add 4.4 GB for, overhead? (Tensorflow and loaded images?)
            print(f"Minimum Memory Requirements: {generator_size + discriminator_size + 4.4} GB")
            
            exit()

        # self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        # self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)

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
            logging.info("Restored from {}".format(self.checkpoint_manager.latest_checkpoint))
        else:
            logging.info("Initializing from scratch.")

        # pre_trained_model = keras.models.load_model("./saved_models/flower_photos")
        # self.discriminator.build(input_shape=self.seed.shape)
        # self.discriminator.layers[3].set_weights(pre_trained_model.layers[4].get_weights())
        # self.discriminator.layers[5].set_weights(pre_trained_model.layers[6].get_weights())
        # self.discriminator.layers[7].set_weights(pre_trained_model.layers[8].get_weights())

    def make_some_noise(self):
        return tf.random.normal([self.batch_size, self.image_height, self.image_width, self.image_depth])

    def make_generator_model(self):
        input_shape = (self.image_height, self.image_width, self.image_depth)
        img_input = keras.Input(shape=input_shape)

        #=> 32 * 32 * 3 = 3,072
        x = layers.experimental.preprocessing.Resizing(32, 32, interpolation="bilinear")(img_input)
        x = layers.Flatten()(x)
                
        #=> 32 * 32 * 3 = 3,072
        x = layers.Dense(32 * 32 * 3, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        
        #=> 36 * 36 * 4 = 5,184
        x = layers.Dense(32 * 32 * 4, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        
        #=> 45 * 45 * 16 = 32400
        x = layers.Dense(45 * 45 * 16, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        x = layers.Reshape((45, 45, 16), input_shape=(45 * 45 * 16,))(x)

        #=> 45 * 45 * 256 = 518,400
        x = layers.Conv2DTranspose(256, 3, strides=(1, 1), activation="relu", padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(256, 3, activation="relu", padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.GaussianNoise(0.1)(x)
        x = layers.LeakyReLU()(x)

        #=> 90 * 90 * 128 = 1,036,800
        x = layers.Conv2DTranspose(128, 5, strides=(2, 2), activation="relu", padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.GaussianNoise(0.1)(x)
        x = layers.LeakyReLU()(x)

        #=> 180 * 180 * 64 = 2,073,600
        x = layers.Conv2DTranspose(64, 5, strides=(2, 2), activation="relu", padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.GaussianNoise(0.1)(x)
        x = layers.LeakyReLU()(x)

        # #=> 360 * 360 * 3 = 388,800
        # x = layers.Conv2DTranspose(3, 5, strides=(2, 2), activation="sigmoid", padding="same", use_bias=False)(x)
        # x = layers.experimental.preprocessing.Rescaling(255)(x)

        #=> 360 * 360 * 3 = 388,800
        x = layers.Conv2DTranspose(3, 5, strides=(2, 2), activation="tanh", padding="same", use_bias=False)(x)
        x = layers.experimental.preprocessing.Rescaling(127.5)(x)
        x = layers.experimental.preprocessing.Rescaling(1, offset=127.5)(x)

        return tf.keras.models.Model(img_input, x, name="generator")

    def make_discriminator_model(self):
        input_shape = (self.image_height, self.image_width, self.image_depth)
        img_input = keras.Input(shape=input_shape)

        # x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(img_input)
        
        #=> 360 * 360 * 3 = 388,800
        x = layers.experimental.preprocessing.Rescaling(1.0, offset=-127.5)(img_input)
        x = layers.experimental.preprocessing.Rescaling(1.0 / 127.5)(x)

        #=> 180 * 180 * 16 = 518,400
        x = layers.Conv2D(16, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D()(x)
        #=> 90 * 90 * 64 = 518,400
        x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D()(x)
        #=> 45 * 45 * 128 = 259,200
        x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D()(x)
        
        #=> 45 * 45 * 16 = 32,400
        x = layers.Conv2D(32, 1, padding="same", activation="relu")(x)

        x = layers.Dropout(0.2)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(128)(x)
        x = layers.Dense(1)(x)

        return tf.keras.models.Model(img_input, x, name="discriminator")

#     def calculate_gradient_penalty_a(self, real_images, fake_images):
#         """ Calculate gradient penalty using ?
#         
#         https://keras.io/examples/generative/wgan_gp/
#         """ 
#         # Get the interpolated image
#         alpha = tf.random.normal([self.batch_size, 1, 1, 1], 0.0, 1.0)
#         diff = fake_images - real_images
#         interpolated = real_images + alpha * diff
# 
#         with tf.GradientTape() as gp_tape:
#             gp_tape.watch(interpolated)
#             # 1. Get the discriminator output for this interpolated image.
#             pred = self.discriminator(interpolated, training=True)
# 
#         # 2. Calculate the gradients w.r.t to this interpolated image.
#         grads = gp_tape.gradient(pred, [interpolated])[0]
#         # 3. Calculate the norm of the gradients.
#         norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
#         gp = tf.reduce_mean((norm - 1.0) ** 2)
#         return gp
# 
#     def calculate_gradient_penalty_b(self, real_images, fake_images):  
#         """ Calculate gradient penalty using WGAN-GP
#
#         https://arxiv.org/abs/1704.00028
#         """
#         shape = [tf.shape(real_images)[0]] + [1] * (real_images.shape.ndims - 1)
#         alpha = tf.random.uniform(shape=shape, minval=0., maxval=1.)
#         inter = real_images + alpha * (fake_images - real_images)
#         inter.set_shape(real_images.shape)
#     
#         with tf.GradientTape() as t:
#             t.watch(inter)
#             pred = self.discriminator(inter, training=True)
#         grad = t.gradient(pred, inter)
#         norm = tf.norm(tf.reshape(grad, [tf.shape(grad)[0], -1]), axis=1)
#         gp = tf.reduce_mean((norm - 1.)**2)
#         return gp

    def calculate_gradient_penalty(self, real_images, _fake_images):
        """ Calculate gradient penalty using DRAGAN
        
        https://arxiv.org/abs/1705.07215v5
        """
        beta = tf.random.uniform(shape=tf.shape(real_images), minval=0., maxval=1.)
        b = real_images + 0.5 * tf.math.reduce_std(real_images) * beta
        shape = [tf.shape(real_images)[0]] + [1] * (real_images.shape.ndims - 1)
        alpha = tf.random.uniform(shape=shape, minval=0., maxval=1.)
        inter = real_images + alpha * (b - real_images)
        inter.set_shape(real_images.shape)

        with tf.GradientTape() as t:
            t.watch(inter)
            pred = self.discriminator(inter, training=True)
        grad = t.gradient(pred, inter)
        norm = tf.norm(tf.reshape(grad, [tf.shape(grad)[0], -1]), axis=1)
        gp = tf.reduce_mean((norm - 1.)**2)
        return gp

    def discriminator_loss(self, real_output, fake_output):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, fake_output):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        return cross_entropy(tf.ones_like(fake_output), fake_output)

    def generate_and_save_images(self, model, input, file_name, print_multiple=False):
        # Notice `training` is set to False.
        # This is so all layers run in inference mode (batchnorm).
        predictions = model(input, training=False)

        preds = predictions[0, :, :, :].numpy()
        logging.debug(f"Results range: {np.min(preds)} - {np.max(preds)}")

        if print_multiple:
            fig = plt.figure(figsize=(6, 6))
            for i in range(predictions.shape[0]):
                plt.subplot(3, 3, i + 1)
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
    def train_step(self, images, gradient_penalty_weight=1.0):
        noise = self.make_some_noise()

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)
            # self.generator.summary()
            
            real_output = self.discriminator(images[0], training=True)
            fake_output = self.discriminator(generated_images, training=True)
            
            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)
            
            gradient_penalty = self.calculate_gradient_penalty(images[0], generated_images)    
            disc_loss_total = disc_loss + gradient_penalty * gradient_penalty_weight
            
            gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(disc_loss_total, self.discriminator.trainable_variables)
    
            self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
            self.discriminator_optimizer.apply_gradients(
                zip(gradients_of_discriminator, self.discriminator.trainable_variables)
            )

            return gen_loss, disc_loss_total

    def train(self, dataset=None, epochs=None):
        ##
        ## WARNING: This will convert RGBA images to RGB (the default setting)
        ##          however, some RGBA images will come out black !!!!!!!!!!
        ##          If you see unexpected large black blobs in the generated
        ##          images, then check if you are using RGBA images and manually
        ##          remove the alpha channel permanently (otside of here)
        ##
        if dataset is None:
            dataset = tf.keras.preprocessing.image_dataset_from_directory(
                self.source_images_path,
                image_size=(self.image_height, self.image_width),
                batch_size=self.batch_size,
            )
            # print(dataset)
            AUTOTUNE = tf.data.experimental.AUTOTUNE
            dataset = dataset.shuffle((self.batch_size * 4), reshuffle_each_iteration=True)

        if epochs is None:
            epochs = self.epochs

        gen_loss = None
        disc_loss = None
        final_gen_loss = 0.0
        final_disc_loss = 0.0

        for epoch in range(epochs):
            start = time.time()

            bar = FillingCirclesBar(
                f"Epoch {epoch}/{epochs} Loss: gen {0.0:7.5f}, disc {0.0:7.5f}",
                max=len(dataset),
            )
            for image_batch in dataset:
                batch_size = image_batch[0].get_shape()[0]
                if batch_size == self.batch_size:
                    gen_loss, disc_loss = self.train_step(image_batch)
                    final_gen_loss = gen_loss.numpy()
                    final_disc_loss = disc_loss.numpy()
                    bar.message = (
                        f"Epoch {epoch}/{epochs} Loss: gen {gen_loss.numpy():7.5f}, disc {disc_loss.numpy():7.5f}"
                    )
                bar.next()
            bar.finish()

            # Produce images for the GIF as we go
            display.clear_output(wait=True)
            image_file_name = f"e{(epoch + 1):04d}-g{final_gen_loss:4.2f}-d{final_disc_loss:4.2f}.png"
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
            
            if np.isnan(gen_loss.numpy()) or np.isnan(disc_loss.numpy()):
                logging.error(f"Unrecoverable error, loss value is not a number")
                break

        # Generate after the final epoch
        display.clear_output(wait=True)
        image_file_name = f"e{(epoch + 1):04d}-g{final_gen_loss:4.2f}-d{final_disc_loss:4.2f}.png"
        self.generate_and_save_images(
            self.generator,
            self.seed,
            os.path.join(self.target_images_path, image_file_name),
            True,
        )

    def draw(self, filename=None):
        if filename is None:
            self.generate_and_save_images(self.generator, self.make_some_noise(), self.target_image_path)
        else:
            self.generate_and_save_images(self.generator, self.make_some_noise(), filename)

    def save(self):
        if self.saved_model_dir is not None:
            self.generator.save(self.saved_model_dir)
        else:
            logging.error("Source model directory not defined")
