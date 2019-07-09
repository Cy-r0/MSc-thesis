import math
import multiprocessing
import os

from imageio import imread, imwrite
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
		- imageset_f (string): path of file that contains list of images to convert.
		- watershed_dir (string): path of directory with transformed images.
		- img_extension (string, optional): default is .png,
		  which is compatible with the PASCAL VOC images.

	"""

	def __init__(self,
				 img_dir,
				 imageset_f,
				 watershed_dir,
				 img_extension = ".png"):
		self.img_dir = img_dir
		self.imageset_f = imageset_f
		self.watershed_dir = watershed_dir
		self.img_ext = img_extension

		if not os.path.isdir(img_dir):
			raise RuntimeError("Image directory not found or corrupted.")
		if not os.path.exists(imageset_f):
			raise RuntimeError("Imageset file not found.")
		
		# Get image names
		with open(os.path.join(imageset_f), "r") as f:
			self.images = [x.strip() for x in f.readlines()]
		
		assert len(self.images) > 0, "No images were retrieved."

	def generate(self,
				 level_spans = [1,2,2,3,3,4,4,5,6,8,11,15,20],  # sum of spans is 84
				 jobs = 4):
		"""
		Generate transformed images using multiple processes.

		Args:
			- level_spans (int, sequence): list of the widths of each
				energy level in pixels.
		
		WARNING!
			Maximum sum of level spans is around 130 on my machine (32 GB RAM)
			before I start getting MemoryErrors.
		"""
		filter = _WatershedFilter(level_spans)
		
		pool = multiprocessing.Pool(processes = jobs)
		for image in self.images:
			pool.apply_async(self._work, args = (filter, image))
		pool.close()
		pool.join()

		print("DONE!")

	def _work(self, filter, image_name):
		"""
		Internal function that converts images.

		Args:
			- filter (class): filter to apply.
			- image_name (string): name of input image.
		"""
		# skip if already exists
		if os.path.exists(os.path.join(self.watershed_dir,
										image_name + self.img_ext)):
			print(image_name + " already exists. Skipped.")
		
		# Read image, process it and save output
		else:
			img = imread(os.path.join(self.img_dir,
									  image_name + self.img_ext))
			transformed = filter.process(img)
			imwrite(os.path.join(self.watershed_dir,
								 image_name + self.img_ext), transformed)
			print(image_name + " transformed.")


class _WatershedFilter(object):
	"""
	Calculates energy level of image pixels depending on their distance to the
	closest mask boundary pixel.
	NB: energy level is just the index of the elements in level_spans

	Args:
		- level_spans (int, sequence): list of the widths of each energy level
			in pixels. Number of energy levels is inferred from here,
			and it is equal to len(level_spans) + 1.
	"""

	def __init__(self, level_spans):
		self.level_spans = level_spans
		self.kernel_halfsize = sum(level_spans)	#centre pixel excluded obv
		self.kernel_size = 2 * self.kernel_halfsize + 1

		self.kernel = np.zeros((self.kernel_size, self.kernel_size), dtype="float")
		# Compose kernel with rounded distance of each pixel from the centre
		for i in range(self.kernel_size):
			for j in range(self.kernel_size):
				self.kernel[i,j] = math.sqrt((i - self.kernel_halfsize)**2 +
											 (j - self.kernel_halfsize)**2)
											  
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

	def _filter(self, array, boundary_colour = 255):
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

		# Return highest energy level if no boundary pixels detected
		if boundary_px[0].size == 0:
			energy_level = len(self.level_spans)

		# Get minimum distance from boundary pixels and convert to energy level
		else:
			min_dist = min([self.kernel[i,j] for i,j
								in zip(boundary_px[0], boundary_px[1])])
			
			cumulative_level = 0
			for i in range(len(self.level_spans)):
				cumulative_level += self.level_spans[i]

				if (cumulative_level < min_dist 
					and i < len(self.level_spans) - 1):
					continue

				elif cumulative_level >= min_dist:
					energy_level = i
					break
				
				else:
					energy_level = i + 1

		return energy_level


# Test code
if __name__ == "__main__":

	root = "/home/cyrus/Datasets/VOCdevkit/VOC2012/"

	w = Watershed(root + "SegmentationClassAug",
				  root + "ImageSets/Segmentation/trainval.txt",
				  root + "WatershedTransform")
	w.generate()