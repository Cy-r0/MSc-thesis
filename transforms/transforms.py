"""
These transforms apply the exact same operation (even random transforms) to both
the input image and the targets (only accepts two targets at a time).
"""

import numbers
import random

from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as F


class ColourJitter(object):
    """
    Randomly change brightness, contrast, saturation and hue of input image only.
    """

    def __init__(self, brightness, contrast, saturation, hue):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, sample):
        img, seg, dist, grad = sample["image"], sample["seg"], sample["dist"], sample["grad"]
        
        img = T.ColorJitter(
            self.brightness,
            self.contrast,
            self.saturation,
            self.hue)(img)

        return {"image": img, "seg": seg, "dist": dist, "grad": grad}


class Normalise(object):
    """
    Normalises only the image (not the targets) according to mean and std.

    Args:
        - mean (sequence): sequence of means for each channel.
        - std (sequence): sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace
    
    def __call__(self, sample):
        img, seg, dist, grad = sample["image"], sample["seg"], sample["dist"], sample["grad"]

        img = F.normalize(img, self.mean, self.std, self.inplace)

        return {"image": img, "seg": seg, "dist": dist, "grad": grad}


class Quantise(object):
    """
    Quantizes pixel values.
    Should only be applied to the distance transformed target.

    Args:
        - level_widths (sequence): width of each quantisation level, starting
            from level 0 onwards. One more level will be automatically added
            at the end, so the number of levels will be len(level_widths) + 1.
    """

    def __init__(self, level_widths):
        assert sum(level_widths) < 256
        self.lookup_table = [len(level_widths)] * 256
        acc = 0

        # Generate lookup table for quantisation
        for i in range(len(level_widths)):
            self.lookup_table[acc : acc + level_widths[i]] = [i] * level_widths[i]
            acc += level_widths[i]

    def __call__(self, sample):
        img, seg, dist, grad = sample["image"], sample["seg"], sample["dist"], sample["grad"]

        dist = dist.point(self.lookup_table)

        return {"image": img, "seg": seg, "dist": dist, "grad": grad}


class RandomHorizontalFlip(object):
    """
    Randomly flip image and targets horizontally (left-right).

    Args:
        - p (int): probability of flipping image.
    """
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, sample):
        img, seg, dist, grad = sample["image"], sample["seg"], sample["dist"], sample["grad"]

        if random.random() > self.p:
            img = F.hflip(img)
            seg = F.hflip(seg)
            dist = F.hflip(dist)
            grad = F.hflip(grad)

        return {"image": img, "seg": seg, "dist": dist, "grad": grad}


class RandomResizedCrop(object):
    """
    Randomly crop both image and targets and rescale output figure
    to desired size. NB: ratio is always 1, so all images will be square.

    Args:
        - scale (tuple): range of scale factors to apply
            (only the extremes need be included in the tuple).
        - size (int or tuple): output size in pixels.
        - ratio (tuple): range of aspect ratios of the cropped regions.
    """
    def __init__(self, scale, size, ratio=(1., 1.)):
        self.scale = scale
        self.size = size
        self.ratio = ratio
    
    def __call__(self, sample):
        img, seg, dist, grad = sample["image"], sample["seg"], sample["dist"], sample["grad"]
        i, j, h, w = T.RandomResizedCrop.get_params(img, self.scale, self.ratio)

        img = F.resized_crop(img, i, j, h, w, self.size, Image.BILINEAR)
        seg = F.resized_crop(seg, i, j, h, w, self.size, Image.NEAREST)
        dist = F.resized_crop(dist, i, j, h, w, self.size, Image.NEAREST)
        grad = F.resized_crop(grad, i, j, h, w, self.size, Image.NEAREST)

        return {"image": img, "seg": seg, "dist": dist, "grad": grad}


class Resize(object):
    """
    Resize both image and targets.
    If size is int, smaller edge is resized to match size, while bigger edge
    is scaled down so that the aspect ratio of the figure doesnt change.

    Args:
        - size (sequence or int): if int, size to rescale smaller edge to.

    NB: It's necessary that the interpolation is NEAREST for the targets,
    otherwise new classes might be created that have no real meaning.
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        img, seg, dist, grad = sample["image"], sample["seg"], sample["dist"], sample["grad"]

        img = F.resize(img, self.size, Image.BILINEAR)
        seg = F.resize(seg, self.size, Image.NEAREST)
        dist = F.resize(dist, self.size, Image.NEAREST)
        grad = F.resize(grad, self.size, Image.NEAREST)

        return {"image": img, "seg": seg, "dist": dist, "grad": grad}


class ToTensor(object):
    """
    Convert PIL Image or ndarray to tensor.
    """

    def __call__(self, sample):
        img, seg, dist, grad = sample["image"], sample["seg"], sample["dist"], sample["grad"]

        img = F.to_tensor(img)
        seg = F.to_tensor(seg)
        dist = F.to_tensor(dist)
        grad = F.to_tensor(grad)

        return {"image": img, "seg": seg, "dist": dist, "grad": grad}
