"""
These transforms apply the exact same operation (even random transforms) to both
the input image and the targets (only accepts two targets at a time).
"""

from collections.abc import Iterable as Iterable
import numbers

from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as F


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

        img, seg, dist = sample["image"], sample["seg"], sample["dist"]

        img = F.normalize(img, self.mean, self.std, self.inplace)

        return {"image": img, "seg": seg, "dist": dist}


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

        for i in range(len(level_widths)):

            self.lookup_table[acc : acc + level_widths[i]] = [i] * level_widths[i]
            acc += level_widths[i]

    def __call__(self, sample):

        img, seg, dist = sample["image"], sample["seg"], sample["dist"]
        
        dist = dist.point(self.lookup_table)

        return {"image": img, "seg": seg, "dist": dist}


class RandomCrop(object):
    """
    Randomly crop both image and targets.

    Args:
        - size (sequence or int): size of output image.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
    
    def __call__(self, sample):

        img, seg, dist = sample["image"], sample["seg"], sample["dist"]

        i, j, h, w = T.RandomCrop.get_params(img, output_size=self.size)

        img = F.crop(img, i, j, h, w)
        seg = F.crop(seg, i, j, h, w)
        dist = F.crop(dist, i, j, h, w)

        return {"image": img, "seg": seg, "dist": dist}


class Resize(object):
    """
    Resize both image and targets.

    Args:
        - size (sequence or int): size to rescale to.
        - interpolation (PIL constant): type of interpolation to use.
    """

    def __init__(self, size, interpolation=Image.BILINEAR):

        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)

        self.size = size
        self.interpolation = interpolation

    def __call__(self, sample):

        img, seg, dist = sample["image"], sample["seg"], sample["dist"]

        img = F.resize(img, self.size, self.interpolation)
        seg = F.resize(seg, self.size, self.interpolation)
        dist = F.resize(dist, self.size, self.interpolation)

        return {"image": img, "seg": seg, "dist": dist}


class ToTensor(object):
    """
    Convert PIL Image or ndarray to tensor.
    """

    def __call__(self, sample):

        img, seg, dist = sample["image"], sample["seg"], sample["dist"]

        img = F.to_tensor(img)
        seg = F.to_tensor(seg)
        dist = F.to_tensor(dist)

        return {"image": img, "seg": seg, "dist": dist}
