from os import listdir
from os.path import join, isdir, isfile, splitext

from PIL import Image

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
		- seg_extension (string, optional): default is compatible with PASCAL VOC.

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
		
		assert len(self.images) > 0, "Looks like there's no images in this folder."

	def generate(self, levels=8):
		"""
		Generate transformed images.

		Args:
			- levels (int): number of energy levels to quantise the transform with.
		"""
		for image_name in self.images:

			image = Image.open(join(self.img_dir, image_name + self.img_ext))

			# watershed algorithm here:
			# use convolutional kernel

			# for each pixel in image:
			# 	pass kernel onto it
			#	check which pixels of the kernel are on a boundary
			#	depending on the index of those pixels, set a value to the center pixel

# test code
if __name__ == "__main__":
	w = Watershed("img", "s")
	w.generate()