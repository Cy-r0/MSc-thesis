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
		- img_dir (string): path of directory with input images.
		- seg_dir (string): path of directory with instance segmentations.
		- watershed_dir (string): path of directory with transformed images.
		- img_extension (string, optional): default is compatible with PASCAL VOC.
		- seg_extension (string, optional): default is compatible with PASCAL VOC.

	"""

	def __init__(self,
				 img_dir,
				 seg_dir,
				 watershed_dir,
				 img_extension = ".jpg",
				 seg_extension = ".png"):
		self.img_dir = img_dir
		self.seg_dir = seg_dir
		self.watershed_dir = watershed_dir
		self.img_ext = img_extension
		self.seg_ext = seg_extension

		if not isdir(img_dir):
			raise RuntimeError("Image directory not found or corrupted.")
		
		# get sorted names of all images and segmentations
		self.images = sorted([splitext(img)[0] for img in listdir(img_dir)
					  if isfile(join(img_dir, img))])
		self.segs = sorted([splitext(seg)[0] for seg in listdir(seg_dir)
					if isfile(join(seg_dir, seg))])
		
		assert len(self.images) == len(self.segs), \
				   "Number of images doesn't match number of segmentation files."

	def generate(self, levels=8):
		"""
		Generate transformed images.

		Args:
			- levels (int): number of energy levels to quantise the transform with.
		"""
		for image_name, seg_name in zip(self.images, self.segs):
			assert image_name == seg_name, \
				   "Image and segmentation file names don't match."

			image = Image.open(join(self.img_dir, image_name + self.img_ext))
			seg = Image.open(join(self.seg_dir, seg_name + self.seg_ext))

			#watershed algorithm here


#test code
if __name__ == "__main__":
	w = Watershed("img", "seg", "s")
	w.generate()