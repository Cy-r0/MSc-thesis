import os

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class VOCDistance(Dataset):
    """
    Dataset based on PASCAL VOC 2012 (assumes it has already been downloaded).
    It uses rgb as samples and instance watershed transform as labels.

    Arguments:
        - root (string): root directory of dataset (devkit).
        - image_set (string): choose from "train", "val" and "trainval".
        - transforms (callable): transforms that take both sample and target
            as input.
    """

    def __init__(self,
                 root,
                 image_set="train",
                 transform=None):
        self.image_set = image_set
        self.transform = transform
        voc_root = os.path.join(root, "VOCdevkit/VOC2012")
        image_dir = os.path.join(voc_root, "JPEGImages")
        target_dir = os.path.join(voc_root, "DistanceTransform")

        if not os.path.isdir(voc_root):
            raise RuntimeError("Dataset not found or corrupted.")

        imageset_dir = os.path.join(voc_root, "ImageSets/Segmentation")
        imageset_file = os.path.join(imageset_dir, image_set.rstrip('\n') + ".txt")

        if not os.path.exists(imageset_file):
            raise ValueError(
                "Wrong image_set entered!")

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
        sample = {'image': img, 'target': target}

        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample

    def __len__(self):
        return len(self.images)