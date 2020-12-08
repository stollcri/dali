#!/usr/bin/env python3
# -*- coding: utf-8 -*-

path = "./generator_images/"
name = ""

def create_mp4(path, name):
	file_list = []
	for file in os.listdir(path):
		if file.startswith(name):
			complete_path = path + file
			file_list.append(complete_path)
	
	file_list.sort()
	
	writer = imageio.get_writer(f"{name}.mp4", fps=20)
	for im in file_list:
		writer.append_data(imageio.imread(im))
	writer.close()

if __name__ == "__main__":
	create_mp4(path, name)