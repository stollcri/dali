IMAGE_SQUARE_SIZE := 720
TARGET_DATASET := line_faces

clean:
	rm -f ./generator_images_${TARGET_DATASET}/*.jpg
	rm -f ./generator_images_${TARGET_DATASET}/*.png
	rm -f ./generator_images_${TARGET_DATASET}/*.mp4
	rm -f ./generator_checkpoints_${TARGET_DATASET}/c*

generate:
	mkdir -p generator_checkpoints_${TARGET_DATASET}
	mkdir -p generator_images_${TARGET_DATASET}
	./generate_${IMAGE_SQUARE_SIZE}.py -v \
	--source-dir ./source_images/${TARGET_DATASET}/ \
	--checkpoint-dir ./generator_checkpoints_${TARGET_DATASET} \
	--target-dir ./generator_images_${TARGET_DATASET}
