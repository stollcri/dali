#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import imageio


def create_mp4(path, name):
	file_list = []
	for index, file in enumerate(os.listdir(path)):
		if index % 8 == 0 and file.startswith(name):
			complete_path = os.path.join(path, file)
			file_list.append(complete_path)
	
	file_list.sort()
	
	video_path = os.path.join(path, f"{name}.mp4")
	writer = imageio.get_writer(video_path, fps=20)
	for im in file_list:
		print(im)
		writer.append_data(imageio.imread(im))
	writer.close()

if __name__ == "__main__":
	create_mp4("./unicode_emojis_generator_images2", "seed")
	# create_mp4("./generator_images/", "sunflower_seed")
