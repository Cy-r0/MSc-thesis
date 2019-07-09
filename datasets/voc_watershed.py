import os

import numpy as np
from PIL import Image
import torch
from torchvision.datasets.vision import VisionDataset
import torchvision.transforms as T


class VOCWatershed(VisionDataset):
    """
    Dataset based on PASCAL VOC 2012 (assumes it has already been downloaded).
    It uses rgb as samples and instance watershed transform as labels.

    Arguments:
        - root (string): root directory of dataset (devkit).
        - image_set (string): choose from "train", "val" and "trainval".
        - transform (callable): transform to apply to samples.
        - target_transform (callable): transform to apply to labels.
        - transforms (callable): transform to apply to both samples and labels.
    """


    def __init__(self,
                 root,
                 image_set = "train",
                 transform = None,
                 target_transform = None,
                 transforms = None):
        super(VOCWatershed, self).__init__(root, transforms, transform, target_transform)
        self.image_set = image_set
        voc_root = os.path.join(self.root, "VOCdevkit/VOC2012")
        image_dir = os.path.join(voc_root, "JPEGImages")
        target_dir = os.path.join(voc_root, "WatershedTransform")

        if not os.path.isdir(voc_root):
            raise RuntimeError("Dataset not found or corrupted.")

        imageset_dir = os.path.join(voc_root, "ImageSets/Segmentation")
        imageset_file = os.path.join(imageset_dir, image_set.rstrip('\n') + ".txt")

        if not os.path.exists(imageset_file):
            raise ValueError(
                'Wrong image_set entered! Please use image_set="train" '
                'or image_set="trainval" or image_set="val"')

        with open(os.path.join(imageset_file), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.targets = [os.path.join(target_dir, x + ".png") for x in file_names]
        
        assert len(self.images) == len(self.targets)
    

    def __getitem__(self, index):
        """
        Arguments:
            - index (int).

        Returns:
            - tuple: image (PIL image), target (PIL Image).
        """
        img = Image.open(self.images[index])
        target = Image.open(self.targets[index])

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        
        return img, target


    def __len__(self):
        return len(self.images)