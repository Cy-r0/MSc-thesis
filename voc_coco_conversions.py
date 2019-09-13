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

from config.config import VOCConfig


cfg = VOCConfig()

class COCO2VOCLabels(object):
    """
    Read a json of predictions in COCO format and change category ids so that 
    classifications match VOC class numbers.
    This class allows to test a COCO-trained model on the VOC dataset without 
    performing any layer surgery.

    Args:
        - coco_json (string): path to json of coco style results.
        - voc_json (string): path to new json of voc class results.
    """

    def __init__(self, coco_json, voc_json):
        self.coco_json = coco_json
        self.voc_json = voc_json

        if not os.path.exists(self.coco_json):
            raise RuntimeError("Invalid path to coco json.")
        
        with open(self.coco_json, "r+") as f:
            self.data_list = json.load(f)
            
        for prediction in self.data_list:

            voc_cat = self._convert(prediction["category_id"])
            prediction["category_id"] = voc_cat

        self.filtered_data_list = [
            pred
            for pred in self.data_list
            if pred["category_id"] is not None
        ]
        
        with open(self.voc_json, "w") as f:
            json.dump(self.filtered_data_list, f)


    def _convert(self, coco_cat_id):
        """
        Convert coco indices to voc indices. Unfortunately the mapping doesn't
        follow any logic, so it needs to be done manually.
        """
        map = {
        # coco: voc
            5 : 1,
            2 : 2,
            15: 3,
            9 : 4,
            40: 5,
            6 : 6,
            3 : 7,
            16: 8,
            57: 9,
            20: 10,
            61: 11,
            17: 12,
            18: 13,
            4 : 14,
            1 : 15,
            59: 16,
            19: 17,
            58: 18,
            7 : 19,
            63: 20,
        }

        if not coco_cat_id in map:
            voc_cat_id = None
        else:
            voc_cat_id = map[coco_cat_id]

        return voc_cat_id


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
        self.unlabelled_colour = np.array([224,224,192])

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
        for k, v in cfg.CLASSES.items():
            dataset_dict = {
                "id": k,
                "name": v
            }
            dataset["categories"].append(dataset_dict)
        dataset["images"] = []
        dataset["annotations"] = []

        ann_i = 0
        for image in tqdm(self.images):

            image_id = int(image.replace("_", ""))

            semantic = cv2.imread(os.path.join(self.sem_dir, image + ".png"))
            instance = cv2.imread(os.path.join(self.inst_dir, image + ".png"))

            # Update list of images
            image_dict = {
                "file_name": image + ".jpg",
                "height": semantic.shape[0],
                "width": semantic.shape[1],
                "id": image_id
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
                        "image_id": image_id,
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
    imageset = "val"

    voc2coco = False

    if voc2coco:
        w = VOC2COCO(root + "SegmentationClassAug",
                    root + "SegmentationObject",
                    root + "ImageSets/Segmentation/" + imageset+".txt",
                    root + "Annotations/pascal_"+imageset+"2012.json")
    else:
        c = COCO2VOCLabels(
            "/home/cyrus/Projects/maskrcnn-benchmark/inference/voc_2012_val_cocostyle/segm.json",
            "/home/cyrus/Projects/maskrcnn-benchmark/inference/voc_2012_val_cocostyle/segm2.json")
    