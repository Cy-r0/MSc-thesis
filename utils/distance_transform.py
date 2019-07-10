import math
import multiprocessing
import os
import timeit

import cv2
import numpy as np


class DistanceTransform(object):
	"""
	Take a directory of pictures and generate a	new directory
	with distance transformed pictures.
	The distance transform converts the image to a representation where
	all the pixel values represent the distance of that pixel from the closest
	boundary pixel.

	Args:
		- img_dir (string): path of directory with instance segmentation images.
		- imageset_f (string): path of file that contains list of images to convert.
		- transformed_dir (string): path of directory with transformed images.
		- img_extension (string, optional): default is .png,
			which is compatible with the PASCAL VOC images.
	"""

	def __init__(self,
				 img_dir,
				 imageset_f,
				 transformed_dir,
				 img_extension=".png"):
		self.img_dir = img_dir
		self.imageset_f = imageset_f
		self.transformed_dir = transformed_dir
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
				 n_jobs):
		"""
		Generate transformed images using multiple processes.

		Args:
			- n_jobs (int): number of processes to spawn.
		"""
		start = timeit.default_timer()

		pool = multiprocessing.Pool(processes=n_jobs)
		for image in self.images:
			pool.apply_async(self._work,
							 args=(image,))
		pool.close()
		pool.join()

		elapsed = timeit.default_timer() - start

		print("Finished! Time taken: %.2f s or %.2f min or %.2f h.\n"
			  %(elapsed, elapsed/60, elapsed/3600))

	def _work(self, image_name):
		"""
		Internal function that converts images.

		Args:
			- image_name (string): name of input image.
		"""
		# skip if already exists
		if os.path.exists(os.path.join(self.transformed_dir,
										image_name + self.img_ext)):
			pass

		else:
			img = cv2.imread(os.path.join(self.img_dir,
										  image_name + self.img_ext),
							 cv2.IMREAD_GRAYSCALE)

			_, thresholded = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
			transformed = cv2.distanceTransform(thresholded, cv2.DIST_L2, 5).astype(int)

			cv2.imwrite(os.path.join(self.transformed_dir,
								 image_name + self.img_ext), transformed)
		
		print("Progress: %d/%d images..."
			  %(len(os.listdir(self.transformed_dir)), len(self.images)),
			  end="\r", flush=True)


# Test code
if __name__ == "__main__":

	root = "/home/cyrus/Datasets/VOCdevkit/VOC2012/"

	w = DistanceTransform(root + "SegmentationClassAug",
				  		  root + "ImageSets/Segmentation/trainval.txt",
				  		  root + "DistanceTransform")
	w.generate(n_jobs=8)