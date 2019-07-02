import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torchvision.datasets.vision import VisionDataset
import torchvision.transforms as T


class VOCClass2Object(VisionDataset):
    """
    Nonstandard dataset class based on PASCAL VOC 2012 (assumes it has already been downloaded).
    It uses class segmentation concatenated to rgb as samples and object segmentation as labels.

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
        super(VOCClass2Object, self).__init__(root, transforms, transform, target_transform)
        self.image_set = image_set
        voc_root = os.path.join(self.root, "VOCdevkit/VOC2012")
        image_dir = os.path.join(voc_root, "JPEGImages")
        # NB: the path below points to the augmented dataset only because it has better color encoding,
        # it doesn't actually use the augmented part of the image set
        class_mask_dir = os.path.join(voc_root, "SegmentationClassAug")
        obj_mask_dir = os.path.join(voc_root, "SegmentationObject")

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
        self.class_masks = [os.path.join(class_mask_dir, x + ".png") for x in file_names]
        self.obj_masks = [os.path.join(obj_mask_dir, x + ".png") for x in file_names]
        assert (len(self.images) == len(self.class_masks)
            and len(self.images) == len(self.obj_masks))
    

    def __getitem__(self, index):
        """
        Arguments:
            - index (int).

        Returns:
            - tuple: image (PIL image), target (PIL Image).
        """
        #combine class mask into alpha channel of rgb image
        img = Image.open(self.images[index])
        mask = Image.open(self.class_masks[index])
        img.putalpha(mask)

        #convert to tensor and save
        totensor = T.ToTensor()
        resize = T.Resize((1024, 1024))
        tens = totensor(resize(img))
        plt.imshow(np.transpose(tens, (1,2,0)))
        plt.savefig("tensor.png", transparent=True)

        target = Image.open(self.obj_masks[index])

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        
        return img, target


    def __len__(self):
        return len(self.images)