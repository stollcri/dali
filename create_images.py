#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib
import mplcairo
import os
import random

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
prop = FontProperties(fname='/System/Library/Fonts/Apple Color Emoji.ttc', size=96)

def create_image(path, name, text):
	matplotlib.rcParams.update({'font.size': 22})
	plt.figure(figsize=(1, 1))
	plt.text(0, 0, text, fontproperties=prop)
	plt.axis('off')
	plt.savefig(os.path.join(path,name), bbox_inches='tight')
	plt.close()

if __name__ == "__main__":
	print('Default backend: ' + matplotlib.get_backend()) 
	matplotlib.use("module://mplcairo.macosx")
	print('Backend is now ' + matplotlib.get_backend())

	# emoji_list = ["🐶", "😆", "😬", "🐱", "😠", "😯", "😺", "👽", "💀", "🤡", "🤖", "🐻‍"]
	emoji_list = [
		'\U0001F479',
		'\U0001F47A',
		'\U0001F47B',
		'\U0001F47D',
		'\U0001F47E',
		'\U0001F47F',
		'\U0001F480',
		# '\U0001F48B',
		# '\U0001F48C',
		# '\U0001F493',
		# '\U0001F495',
		# '\U0001F496',
		# '\U0001F497',
		# '\U0001F498',
		# '\U0001F49D',
		# '\U0001F49E',
		# '\U0001F4A9',
		'\U0001F600',
		'\U0001F601',
		'\U0001F602',
		'\U0001F603',
		'\U0001F604',
		'\U0001F605',
		'\U0001F606',
		'\U0001F607',
		'\U0001F608',
		'\U0001F609',
		'\U0001F60A',
		'\U0001F60B',
		'\U0001F60C',
		'\U0001F60D',
		'\U0001F60E',
		'\U0001F60F',
		'\U0001F610',
		'\U0001F611',
		'\U0001F612',
		'\U0001F613',
		'\U0001F614',
		'\U0001F615',
		'\U0001F616',
		'\U0001F617',
		'\U0001F618',
		'\U0001F619',
		'\U0001F61A',
		'\U0001F61B',
		'\U0001F61C',
		'\U0001F61D',
		'\U0001F61E',
		'\U0001F61F',
		'\U0001F620',
		'\U0001F621',
		'\U0001F622',
		'\U0001F623',
		'\U0001F624',
		'\U0001F625',
		'\U0001F626',
		'\U0001F627',
		'\U0001F628',
		'\U0001F629',
		'\U0001F62A',
		'\U0001F62B',
		'\U0001F62C',
		'\U0001F62D',
		'\U0001F62E',
		'\U0001F62F',
		'\U0001F630',
		'\U0001F631',
		'\U0001F632',
		'\U0001F633',
		'\U0001F634',
		'\U0001F635',
		'\U0001F636',
		'\U0001F637',
		'\U0001F638',
		'\U0001F639',
		'\U0001F63A',
		'\U0001F63B',
		'\U0001F63C',
		'\U0001F63D',
		'\U0001F63E',
		'\U0001F63F',
		'\U0001F640',
		'\U0001F641',
		'\U0001F642',
		'\U0001F643',
		'\U0001F644',
		'\U0001F648',
		'\U0001F649',
		'\U0001F64A',
		'\U0001F910',
		'\U0001F911',
		'\U0001F912',
		'\U0001F913',
		'\U0001F914',
		'\U0001F915',
		'\U0001F916',
		'\U0001F917',
	]
	for idx,text in enumerate(emoji_list):
		for sub_idx in range(0,1):
			create_image("./source_images/emoji_images_unicode/", f"created_image_{idx}_{sub_idx}.jpg", text)
