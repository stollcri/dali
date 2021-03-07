#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
import tensorflow as tf
import time

from IPython import display
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

class DeepConvolutionalGenerativeAdversarialNetwork(object):
    def __init__(self, args):
        self.buffer_size = 60000
        self.batch_size = 32

        self.image_height = 720
        self.image_width = 720
class_names = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]

discriminator = keras.models.load_model("./flowers")
discriminator.summary()

print()

# layers.UpSampling2D(input_shape=(90, 90, 32), interpolation="bilinear"),
# layers.Conv2DTranspose(16, 3, padding="same", activation="relu"),
# layers.UpSampling2D(input_shape=(180, 180, 16), interpolation="bilinear"),
# layers.Conv2DTranspose(3, 3, padding="same", activation="relu"),
# layers.experimental.preprocessing.Rescaling(1.0 * 255),
# new_model = Sequential(
#     [
#         layers.Reshape((45, 45, 64), input_shape=(129600,)),
#         layers.Conv2DTranspose(32, (2, 2), padding="same", activation="relu"),
#     ]
# )
# new_model.summary()


def make_generator_model():
    model = tf.keras.Sequential(
        [
            layers.experimental.preprocessing.Rescaling(1.0 / 255),
            layers.Conv2D(16, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2DTranspose(
                32, (4, 4), strides=(2, 2), padding="same", use_bias=False
            ),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.Conv2DTranspose(
                16, (4, 4), strides=(2, 2), padding="same", use_bias=False
            ),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.Conv2DTranspose(
                3, (4, 4), strides=(2, 2), padding="same", use_bias=False
            ),
        ]
    )
    # model.summary()
    return model


# sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
# sunflower_path = tf.keras.utils.get_file("Red_sunflower", origin=sunflower_url)
sunflower_path = pathlib.Path("./592px-Red_sunflower.jpg")

# img = keras.preprocessing.image.load_img(
#     sunflower_path, target_size=(image_height, image_width)
# )
# img_array = keras.preprocessing.image.img_to_array(img)
# img_array = tf.expand_dims(img_array, 0)
#
generator = make_generator_model()
# generated_image = generator(img_array, training=False)
# generator.summary()
#
# plt.imshow(generated_image[0, :, :, 0])
# plt.show()

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

checkpoint_dir = "./training_checkpoints"
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(
    generator_optimizer=generator_optimizer,
    discriminator_optimizer=discriminator_optimizer,
    generator=generator,
    discriminator=discriminator,
)

EPOCHS = 32
noise_dim = 360 * 360 * 3
num_examples_to_generate = 2

# We will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
# seed = tf.random.normal([num_examples_to_generate, noise_dim])
img = keras.preprocessing.image.load_img(
    sunflower_path, target_size=(image_height, image_width)
)
img_array = keras.preprocessing.image.img_to_array(img)
seed = tf.expand_dims(img_array, 0)

def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        # plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap="gray")
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5)
        plt.axis("off")

    plt.savefig("image_at_epoch_{:04d}.png".format(epoch))
    # plt.show()

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    img_array = keras.preprocessing.image.img_to_array(img)

    # random boolean mask for which values will be changed
    m = np.random.randint(0, 4, size=img_array.shape).astype(np.bool)
    mask = np.invert(m)

    # random matrix the same shape of your data
    r = np.random.rand(*img_array.shape) * np.max(img_array)
    rando = r.astype(int)

    # use your mask to replace values in your input array
    img_array[mask] = rando[mask]
    img_array = tf.expand_dims(img_array, 0)

    noise = img_array
    # noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images[0], training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(
        disc_loss, discriminator.trainable_variables
    )

    generator_optimizer.apply_gradients(
        zip(gradients_of_generator, generator.trainable_variables)
    )
    discriminator_optimizer.apply_gradients(
        zip(gradients_of_discriminator, discriminator.trainable_variables)
    )


def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch)

        # Produce images for the GIF as we go
        display.clear_output(wait=True)
        generate_and_save_images(generator, epoch + 1, seed)

        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print("Time for epoch {} is {} sec".format(epoch + 1, time.time() - start))

    # Generate after the final epoch
    display.clear_output(wait=True)
    generate_and_save_images(generator, epochs, seed)


data_dir = pathlib.Path("/Users/stollcri/.keras/datasets/flower_photos")
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    image_size=(image_height, image_width),
    batch_size=self.batch_size,
)

AUTOTUNE = tf.data.experimental.AUTOTUNE
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

train(train_dataset, EPOCHS)
