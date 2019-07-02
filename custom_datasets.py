import os

from PIL import Image
from torchvision.datasets.vision import VisionDataset


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
        seg_class_dir = os.path.join(voc_root, "SegmentationClass")
        seg_obj_dir = os.path.join(voc_root, "SegmentationObject")

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

        self.images = [os.path.join(image_dir, x + ".png") for x in file_names]
        self.masks = [os.path.join(seg_obj_dir, x + ".png") for x in file_names]
        assert (len(self.images) == len(self.masks))
    

    def __getitem__(self, index):
        """
        Arguments:
            - index (int).

        Returns:
            - tuple: (image, target).
        """
        img = Image.open(self.images[index])
        target = Image.open(self.masks[index])

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        
        return img, target


    def __len__(self):
        return len(self.images)