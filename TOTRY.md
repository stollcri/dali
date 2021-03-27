# Things to Try

- Make the discriminator colorblind, but leave the generator fully color aware, see if interesting color patterns emerge

```python
    def make_discriminator_model(self):
        def color_to_bw_a(x):
            return (0.21 * x[:,:,:,:1]) + (0.72 * x[:,:,:,1:2]) + (0.07 * x[:,:,:,-1:])

        input_shape = (self.image_height, self.image_width, self.image_depth)
        img_input = keras.Input(shape=input_shape)

        x = layers.Lambda(color_to_bw_a)(img_input)

        # x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(img_input)
        
        #=> 360 * 360 * 3 = 388,800
        x = layers.experimental.preprocessing.Rescaling(1.0, offset=-127.5)(x)
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
        
        #=> 45 * 45 * 32 = 32,400
        x = layers.Conv2D(32, 1, padding="same", activation="relu")(x)

        x = layers.Dropout(0.2)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(128)(x)
        x = layers.Dense(1)(x)

        return tf.keras.models.Model(img_input, x, name="discriminator")
```

## Boring ToDo Items

The idea behind making smaller image size generators is to see if they have the capacity to learn a higher number of basic features and thus create a more diverse set of resulting images. My hypotheses is that the limiting factor on creativity is the size of the dense neural network layers. Dense layers are memory intensive, so their size is limited by the amount of RAM available. If dense layer bandwidth is the limiting factor in generator creativity, then a better approach (before switching to a StyleGAN type of approach) would be to make smaller images which feed into another model that is trained to do high resolution image upsizing. 

- Update DCGAN 128 from learnings in DCGAN 360
- Tune DCGAN 128 for its new network architecture
- Update DCGAN 64 from DCGAN 128
- Tune DCGAN 64 for its new network architecture
