# Things to Try

- Make the discriminator colorblind, but leave the generator fully color aware, see if interesting color patterns emerge

## Boring ToDo Items

The idea behind making smaller image size generators is to see if they have the capacity to learn a higher number of basic features and thus create a more diverse set of resulting images. My hypotheses is that the limiting factor on creativity is the size of the dense neural network layers. Dense layers are memory intensive, so their size is limited by the amount of RAM available. If dense layer bandwidth is the limiting factor in generator creativity, then a better approach (before switching to a StyleGAN type of approach) would be to make smaller images which feed into another model that is trained to do high resolution image upsizing. 

- Update DCGAN 128 from learnings in DCGAN 360
- Tune DCGAN 128 for its new network architecture
- Update DCGAN 64 from DCGAN 128
- Tune DCGAN 64 for its new network architecture
