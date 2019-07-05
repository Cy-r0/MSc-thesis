from math import sqrt
from os import listdir
from os.path import join, isdir, isfile, splitext

import numpy as np
from PIL import Image
from scipy.ndimage import generic_filter

class Watershed(object):
	"""
	Take a directory of pictures and generate a	new directory
	with watershed transformed pictures.
	The watershed transform is quantised with N energy levels,
	where level 0 corresponds to instance boundaries and 
	level N corresponds to regions far from boundaries.

	Args:
		- img_dir (string): path of directory with instance segmentation images.
		- watershed_dir (string): path of directory with transformed images.
		- img_extension (string, optional): default is .png,
		  which is compatible with the PASCAL VOC images.

	"""

	def __init__(self, img_dir, watershed_dir, img_extension = ".png"):
		self.img_dir = img_dir
		self.watershed_dir = watershed_dir
		self.img_ext = img_extension

		if not isdir(img_dir):
			raise RuntimeError("Image directory not found or corrupted.")
		
		# get name of all images
		self.images = sorted([splitext(img)[0] for img in listdir(img_dir)
					  if isfile(join(img_dir, img))])
		
		assert len(self.images) > 0, "No images were retrieved."

	def generate(self, level_spans = [2,3,5,8,13,21,34,55]):
		"""
		Generate transformed images.

		Args:
			- level_spans (int, sequence): list of the thickness of each energy level in pixels.
				Number of energy levels is autoinferred from this argument.
		"""
		for image_name in self.images:

			image = Image.open(join(self.img_dir, image_name + self.img_ext))

			self._rounded_distance_filter(level_spans)

			# watershed algorithm here:
			# use scipy's generic_filter

			# for each pixel in image:
			# 	pass kernel onto it
			#	check which pixels of the kernel are on a boundary (white pixels)
			#	depending on the index of those pixels, set a value to the center pixel
	
	def _rounded_distance_filter(self, level_spans):
		"""
		Filter kernel for calculating distance to closest boundary pixel.

		Args:
			- level_spans (int, sequence): same argument as in self.generate().
		"""
		kernel_halfsize = sum(level_spans)
		kernel_size = 2 * kernel_halfsize + 1

		kernel = np.zeros((kernel_size, kernel_size), dtype="int")
		# calculate rounded distance of each pixel from the centre
		for i in range(kernel_size):
			for j in range(kernel_size):
				kernel[i,j] = round(sqrt((i - kernel_halfsize)**2 +
										 (j - kernel_halfsize)**2))


# test code
if __name__ == "__main__":
	w = Watershed("/home/cyrus/Documents/datasets/VOCdevkit/VOC2012/SegmentationClassAug", "s")
	w.generate()