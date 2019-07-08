from math import sqrt
from os import listdir
from os.path import join, isdir, isfile, splitext

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import generic_filter


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

	def generate(self, level_spans = [2,3,4,5,6,8,12]):	#sum of spans is 40
		"""
		Generate transformed images.

		Args:
			- level_spans (int, sequence): list of the widths of each
				energy level in pixels.
		
		WARNING!
			Maximum sum of level spans is around 120 on my machine (32 GB RAM)
			before the pc starts filling the swap space.
			MemoryErrors still don't occur up to around 130 (swap space is 8 GB),
			however I really don't recommend making the filter leak into
			swap space because the machine will be really slow.
		"""
		filter = _WatershedFilter(level_spans)

		for image_name in self.images:
			# read image, process it and save output
			img = plt.imread(join(self.img_dir, image_name + self.img_ext))
			transformed = filter.process(img)
			# TODO: save transformed
			plt.imshow(transformed)
			plt.show()


class _WatershedFilter(object):
	"""
	Calculates energy level of image pixels depending on its distance to the
	closest mask boundary pixel.
	NB: energy level is just the index of the elements in level_spans

	Args:
		- level_spans (int, sequence): list of the widths of each
			energy level in pixels.
	"""

	def __init__(self, level_spans):
		self.level_spans = level_spans
		self.kernel_halfsize = sum(level_spans)	#centre pixel excluded obv
		self.kernel_size = 2 * self.kernel_halfsize + 1

		self.kernel = np.zeros((self.kernel_size, self.kernel_size), dtype="int")
		# calculate rounded distance of each pixel from the centre
		for i in range(self.kernel_size):
			for j in range(self.kernel_size):
				self.kernel[i,j] = round(sqrt((i - self.kernel_halfsize)**2 +
											  (j - self.kernel_halfsize)**2))
											  
	def process(self, img):
		"""
		Process an image as a 2D numpy array and produce filter output.

		Args:
			- img (ndarray): the input image to be processed with the filter.
		
		Returns:
			- output (ndarray): processed image.
		"""
		return generic_filter(img,
							  self._filter,
							  size = self.kernel_size,
							  mode = "nearest")

	def _filter(self, array, boundary_colour = 1.0):
		"""
		Helper method that implements the filter's function.

		Args:
			- array (ndarray): input array.

		Returns:
			- output (int): energy level of pixel.
		"""
		array = array.reshape((self.kernel_size, self.kernel_size))
		boundary_px = np.where(array == boundary_colour)
		energy_level = 0

		# return highest energy level if no boundary pixels detected
		if boundary_px[0].size == 0:
			energy_level = len(self.level_spans) - 1

		# get minimum distance from boundary pixels and convert to energy level
		else:
			min_dist = min([self.kernel[i,j] for i,j
								in zip(boundary_px[0], boundary_px[1])])
			
			cumulative_level = 0
			for i in range(len(self.level_spans)):
				cumulative_level += self.level_spans[i]

				if cumulative_level < min_dist:
					continue

				elif (cumulative_level >= min_dist):
					energy_level = i
			
		print(energy_level)
		return energy_level


# test code
if __name__ == "__main__":
	w = Watershed("/home/cyrus/Datasets/VOCdevkit/VOC2012/SegmentationClassAug", "s")
	w.generate()