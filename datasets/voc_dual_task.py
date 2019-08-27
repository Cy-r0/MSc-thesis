import os

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class VOCDualTask(Dataset):
    """
    Dataset based on PASCAL VOC 2012 (assumes it has already been downloaded).
    It uses rgb as samples and (instance segmentation, distance transform) as labels.

    Arguments:
        - root (string): root directory of dataset (devkit).
        - image_set (string): choose from "train", "val", "trainval" and
            ."train_75perc" and "train_25perc" (generated by me).
        - transforms (callable): transforms that take both sample and targets
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
        seg_dir = os.path.join(voc_root, "SegmentationClassAug")
        dist_dir = os.path.join(voc_root, "DistanceTransform")
        grad_dir = os.path.join(voc_root, "DTGradientDirection")

        if not os.path.isdir(voc_root):
            raise RuntimeError("Dataset not found or corrupted.")

        imageset_dir = os.path.join(voc_root, "ImageSets/Segmentation")
        imageset_file = os.path.join(imageset_dir, image_set.rstrip('\n') + ".txt")

        if not os.path.exists(imageset_file):
            raise ValueError("Wrong image_set entered!")

        with open(os.path.join(imageset_file), "r") as f:
            file_names = [l.strip() for l in f.readlines()]

        self.img_ids = [n.replace("_", "") for n in file_names]
        self.images = [os.path.join(image_dir, n + ".jpg") for n in file_names]
        self.segs = [os.path.join(seg_dir, n + ".png") for n in file_names]
        self.dists = [os.path.join(dist_dir, n + ".png") for n in file_names]
        self.grads = [os.path.join(grad_dir, n + ".png") for n in file_names]
        
        assert (len(self.images) == len(self.segs)
            and len(self.images) == len(self.dists)
            and len(self.images) == len(self.grads))
    
    def __getitem__(self, index):
        """
        Arguments:
            - index (int).

        Returns:
            - indexed_sample (dict) {
                img_id (string): image filename stripped of underscores and ext.
                image (tensor): input image.
                seg (tensor): semantic segmentation.
                dist (tensor): distance representation.
                grad (tensor): gradient direction of distance representation.
            }
        """
        img_id = self.img_ids[index]
        img = Image.open(self.images[index])
        seg = Image.open(self.segs[index])
        dist = Image.open(self.dists[index])
        grad = Image.open(self.grads[index])

        sample = {"image": img, "seg": seg, "dist": dist, "grad": grad}

        if self.transform is not None:
            sample = self.transform(sample)
        
        indexed_sample = {
            "img_id": img_id,
            "image": sample["image"],
            "seg": sample["seg"],
            "dist": sample["dist"],
            # Get rid of blue channel (its empty)
            "grad": sample["grad"][0:2,:,:]
        }

        return indexed_sample

    def __len__(self):
        return len(self.images)