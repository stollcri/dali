IMAGE_SQUARE_SIZE := 1080
TARGET_DATASET := flower_photos_some

clean:
	rm -f ./generator_images_${TARGET_DATASET}/*.jpg
	rm -f ./generator_images_${TARGET_DATASET}/*.png
	rm -f ./generator_images_${TARGET_DATASET}/*.mp4
	rm -f ./generator_checkpoints_${TARGET_DATASET}/c*

generate:
	@mkdir -p generated_checkpoints/${TARGET_DATASET}
	@mkdir -p generated_images/${TARGET_DATASET}
	
	./generate_${IMAGE_SQUARE_SIZE}.py -v \
	--source-dir ./source_images/${TARGET_DATASET}/ \
	--checkpoint-dir ./generated_checkpoints/${TARGET_DATASET} \
	--target-dir ./generated_images/${TARGET_DATASET}
