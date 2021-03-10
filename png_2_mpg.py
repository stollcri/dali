#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import imageio
import logging
import os


def create_mp4(path, name, skip, fps):
	supported_extensions = [".png", ".jpg", ".jpeg"]

	file_list = []
	for index, file in enumerate(os.listdir(path)):
		file_name, file_extension = os.path.splitext(file)
		if file_extension in supported_extensions:
			if index % skip == 0 and file.startswith(name):
				complete_path = os.path.join(path, file)
				file_list.append(complete_path)
	
	file_list.sort()
	
	video_path = os.path.join(path, f"{name}.mp4")
	writer = imageio.get_writer(video_path, fps=fps)
	for im in file_list:
		print(im)
		writer.append_data(imageio.imread(im))
	writer.close()

if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="Create a movie from a series of images"
	)
	parser.add_argument(
		metavar="IMAGE_DIRECTORY",
		help="Directory to read images from",
		dest="image_directory",
		default=None,
	)
	parser.add_argument(
		metavar="IMAGE_PREFIX",
		help="Prefix of image files",
		dest="image_prefix",
		default=None,
	)
	parser.add_argument(
		"-s", "--skip",
		type=int,
		help="The number images to skip",
		dest="skip",
		default=4,
	)
	parser.add_argument(
		"-f", "--fps",
		type=int,
		help="The number of frames per second",
		dest="fps",
		default=24,
	)
	parser.add_argument(
		"-v", "--verbose", help="verbose output", dest="verbose", action="store_true"
	)
	args = parser.parse_args()

	if (args.image_directory is None or args.image_prefix is None):
		parser.print_usage()
		exit()

	if args.verbose:
		logging.basicConfig(format="%(message)s", level=logging.DEBUG)
	else:
		logging.basicConfig(format="%(message)s", level=logging.INFO)

	create_mp4(
		args.image_directory,
		args.image_prefix,
		args.skip,
		args.fps
	)