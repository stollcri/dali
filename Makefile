clean:
	rm -f ./generator_images/*.png
	rm -f ./generator_images/*.mp4
	rm -f ./generator_checkpoints/c*

generate:
	./generate.py \
	--source-dir ./flower_photos_some/ \
	--checkpoint-dir ./generator_checkpoints \
	--target-dir ./generator_images
