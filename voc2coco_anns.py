from itertools import groupby
import json
import math
import os
import timeit

import cv2
import matplotlib.pyplot as plt
import numpy as np
from pycocotools import mask
from pycocotools.coco import COCO
import skimage.io as io
from tqdm import tqdm

from config.config import VOCSettings


sett = VOCSettings()

class VOC2COCO(object):
	"""
	Take a directory of PASCAL VOC style annotations (i.e. semantic and object
	segmentations), and convert them to COCO style json annotations.

	Args:
		- sem_dir (string): path of directory with semantic segmentation images.
		- inst_dir (string): path of directory with instance segmentation images.
		- imageset_f (string): path of file that contains list of images to convert.
		- anns_f (string): path of new json file containing annotations.
	"""

	def __init__(self,
				 sem_dir,
				 inst_dir,
				 imageset_f,
				 anns_f):
		self.sem_dir = sem_dir
		self.inst_dir = inst_dir
		self.imageset_f = imageset_f
		self.anns_f = anns_f
		self.background_colour = np.array([0, 0, 0])
		self.unlabelled_colour = np.array([192, 224, 224])

		if not os.path.isdir(sem_dir):
			raise RuntimeError("Semantic segm directory not found or corrupted.")
		if not os.path.isdir(inst_dir):
			raise RuntimeError("Instance segm directory not found or corrupted.")
		if not os.path.exists(imageset_f):
			raise RuntimeError("Imageset file not found.")
		
		# Get image names
		with open(os.path.join(imageset_f), "r") as f:
			self.images = [x.strip() for x in f.readlines()]
		
		assert len(self.images) > 0, "No images were retrieved."

		dataset = dict()

		# Populate all fields that can be filled without scanning images
		dataset["info"] = {
			"description": "VOC 2012 instance segmentations",
			"year": 2019,
			"contributor": "Ciro Cursio"
		}
		dataset["categories"] = []
		for k, v in sett.CLASSES.items():
			dataset_dict = {
				"id": k,
				"name": v
			}
			dataset["categories"].append(dataset_dict)
		dataset["images"] = []
		dataset["annotations"] = []

		ann_i = 0
		for i, image in enumerate(tqdm(self.images)):

			semantic = cv2.imread(os.path.join(self.sem_dir, image + ".png"))
			instance = cv2.imread(os.path.join(self.inst_dir, image + ".png"))

			# Update list of images
			image_dict = {
				"file_name": image + ".jpg",
				"height": semantic.shape[0],
				"width": semantic.shape[1],
				"id": i
			}
			dataset["images"].append(image_dict)

			# Update list of segmentations
			colours = np.unique(instance.reshape(-1, instance.shape[2]), axis=0)
			for c in colours:
				
				# Ignore background and unlabelled instances
				if (np.array_equal(c, self.background_colour)
					or np.array_equal(c, self.unlabelled_colour)):
					continue
				else:
					masked = cv2.inRange(instance, c, c)

					# Encode mask to uncompressed RLE
					masked[masked == 255] = 1
					rle = self._mask2rle(masked)

					# Find the other properties of the segmentation
					area = cv2.countNonZero(masked)
					bbox = cv2.boundingRect(cv2.findNonZero(masked))
					masked_semantic = cv2.bitwise_and(semantic, semantic, mask=masked)
					semantic_colours = np.unique(masked_semantic)
					assert len(semantic_colours) == 2 and semantic_colours[0] == 0
					cat_id = int(semantic_colours[1])

					ann_dict = {
						"segmentation": rle,
						"area": area,
						"iscrowd": 0,
						"image_id": i,
						"bbox": bbox,
						"category_id": cat_id,
						"id": ann_i
					}
					dataset["annotations"].append(ann_dict)

					ann_i += 1

		with open(anns_f, 'w') as f:
			json.dump(dataset, f)
		
	def _mask2rle(self, mask):
		"""
		Convert 2d binary mask to run length encoding format.
		"""
		rle = {"counts": [], "size": list(mask.shape)}
		counts = rle.get("counts")
		for i, (value, elements) in enumerate(groupby(mask.ravel(order="F"))):
			if i == 0 and value == 1:
				counts.append(0)
			counts.append(len(list(elements)))
		return rle	


if __name__ == "__main__":

	root = "/home/cyrus/Datasets/VOCdevkit/VOC2012/"
	imageset = "train_reduced"

	if False:
		w = VOC2COCO(root + "SegmentationClassAug",
					root + "SegmentationObject",
					root + "ImageSets/Segmentation/" + imageset+".txt",
					root + "COCOStyleAnnotations/instances_"+imageset+".json")
	
	

	# initialise COCO api
	coco = COCO(root + "COCOStyleAnnotations/instances_"+imageset+".json")

	# display COCO categories and supercategories
	cats = coco.loadCats(coco.getCatIds())
	nms=[cat['name'] for cat in cats]
	print('COCO categories: \n\n', ' '.join(nms))

	# get all images containing given categories, select one at random
	catIds = coco.getCatIds(catNms=['aeroplane','person', "bus"])
	imgIds = coco.getImgIds(catIds=catIds)
	img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]

	# load and display image
	I = io.imread(root + "JPEGImages/" + img['file_name'])
	plt.figure()
	plt.axis('off')
	plt.imshow(I)
	plt.show()

	# load and display instance annotations
	plt.imshow(I); plt.axis('off')
	annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
	anns = coco.loadAnns(annIds)
	coco.showAnns(anns)
	plt.show()
