DALI_IMAGE_SIZE := 1080
TARGET_DATASET := sunflowers

clean:
	rm -f ./generator_images_${TARGET_DATASET}/*.jpg
	rm -f ./generator_images_${TARGET_DATASET}/*.png
	rm -f ./generator_images_${TARGET_DATASET}/*.mp4
	rm -f ./generator_checkpoints_${TARGET_DATASET}/c*

print:
	@mkdir -p generated_checkpoints/${TARGET_DATASET}
	@mkdir -p generated_images/${TARGET_DATASET}
	
	./generate.py -p -v \
	--source-dir ./source_images/${TARGET_DATASET}/ \
	--checkpoint-dir ./generated_checkpoints/${TARGET_DATASET} \
	--target-dir ./generated_images/${TARGET_DATASET}
	
generate:
	@mkdir -p generated_checkpoints/${TARGET_DATASET}
	@mkdir -p generated_images/${TARGET_DATASET}
	
	./generate.py -v \
	--source-dir ./source_images/${TARGET_DATASET}/ \
	--checkpoint-dir ./generated_checkpoints/${TARGET_DATASET} \
	--target-dir ./generated_images/${TARGET_DATASET}
	
draw:
	./draw.py -v \
	./generated_checkpoints/${TARGET_DATASET} \
	./generated_images/${TARGET_DATASET}/${TARGET_DATASET}.jpg
	
movie:
	./png_2_mpg.py ./generated_images/${TARGET_DATASET}/ seed

SAVED_MODEL := sunflowers_3a

draw-saved:
	@mkdir -p saved_models/${SAVED_MODEL}/generated_images
	
	./draw.py -v \
	./saved_models/${SAVED_MODEL}/generated_checkpoint/ \
	./saved_models/${SAVED_MODEL}/generated_images/${TARGET_DATASET}.jpg

save-saved:
	@mkdir -p saved_models/${SAVED_MODEL}/generated_model
	
	./ckpt_2_saved.py -v \
	./saved_models/${SAVED_MODEL}/generated_checkpoint/ \
	./saved_models/${SAVED_MODEL}/generated_model/
