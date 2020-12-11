#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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


image_height = 360
image_width = 360
image_depth = 3

def make_generator_model():
	model = tf.keras.Sequential(
		[
			layers.experimental.preprocessing.Rescaling(
				1.0 / 255,
				input_shape=(image_height, image_width, image_depth),
			),
			
			layers.Conv2D(16, 3, padding="same", activation="relu", trainable=False),
			layers.MaxPooling2D(),
			layers.Conv2D(32, 3, padding="same", activation="relu", trainable=False),
			layers.MaxPooling2D(),
			layers.Conv2D(64, 3, padding="same", activation="relu", trainable=False),
			layers.MaxPooling2D(),
# 			
# 			layers.Conv2DTranspose(
# 				32, (5, 5), strides=(2, 2), padding="same", use_bias=False
# 			),
# 			layers.BatchNormalization(),
# 			layers.LeakyReLU(),
# 
# 			layers.Conv2DTranspose(
# 				16, (5, 5), strides=(2, 2), padding="same", use_bias=False
# 			),
# 			layers.BatchNormalization(),
# 			layers.LeakyReLU(),
# 
# 			layers.Conv2DTranspose(
# 				3,
# 				(5, 5),
# 				strides=(2, 2),
# 				padding="same",
# 				use_bias=False,
# 				activation="sigmoid",
# 			),
# 			layers.BatchNormalization(),
# 			layers.LeakyReLU(),
			
			# 90* 90* 64 = 518400
			# /3 172800
			# layers.Reshape((480, 360, 3)),
			
			# 45*45*64 = 129600
			# /3 43200
			layers.Reshape((240, 180, 3)),
						
			# layers.experimental.preprocessing.Resizing(
			# 	height, width, interpolation='bilinear'
			# )
			
			# layers.UpSampling2D(),
			# layers.Conv2DTranspose(
			# 	3,
			# 	(5, 5),
			# 	strides=(2, 2),
			# 	padding="same",
			# 	use_bias=False,
			# 	activation="sigmoid",
			# ),
			# layers.Activation('sigmoid'),
			layers.experimental.preprocessing.Rescaling(255*16),
		]
	)
	return model
	
generator = make_generator_model()

pre_trained_model = keras.models.load_model("./flower_model")
generator.layers[1].set_weights(pre_trained_model.layers[2].get_weights())
generator.layers[3].set_weights(pre_trained_model.layers[4].get_weights())
generator.layers[5].set_weights(pre_trained_model.layers[6].get_weights())

generator.summary()

noise = tf.random.normal([1, 360, 360, 3])
generated_images = generator(noise, training=False)

preds = generated_images[0, :, :, :].numpy()
print(f"Results range: {np.min(preds)} - {np.max(preds)}")

sizes = np.shape(generated_images[0, :, :, :].numpy().astype("uint8"))
fig = plt.figure(figsize=(1, 1))
ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
ax.set_axis_off()
fig.add_axes(ax)
ax.imshow(
	generated_images[0, :, :, :].numpy().astype("uint8"),
	cmap=plt.get_cmap("bone"),
)
plt.savefig("./experiment.png", dpi=sizes[0])
plt.close()
