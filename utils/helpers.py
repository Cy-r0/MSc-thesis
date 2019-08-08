import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from tensorboardX.utils import figure_to_image
import torch
import torch.nn as nn



def adjust_lr(optimizer, i, max_i, initial_lr, pow):
    """
    Gradually decrease learning rate as iterations increase.
    """
    lr = initial_lr * (1 - i/(max_i + 1)) ** pow
    optimizer.param_groups[0]['lr'] = lr
    optimizer.param_groups[1]['lr'] = 10 * lr
    return lr

def colormap(batch, cmap="viridis"):
    """
    Convert grayscale images to matplotlib colormapped image.

    Args:
        - batch (3D tensor): images to convert.
        - cmap (string): name of colormap to use.
    """
    cmap = plt.cm.get_cmap(name=cmap)

    # Get rid of singleton dimension (n. channels)
    batch = batch.squeeze()

    # Apply colormap and get rid of alpha channel
    batch = torch.tensor(cmap(batch))[..., 0:3]

    # Swap dimensions to match NCHW format
    batch = batch.permute(0, 3, 1, 2)

    return batch

def get_params(model, key):
    """
    Get model parameters.
    NB: backbone is trained 10 times slower than the rest of the network.
    Also, only conv2d layers are trained, batchnormalisation layers are kept
        the same because backbone was already pretrained on Imagenet.
    """
    for m in model.named_modules():
        if key == "1x":
            if "backbone" in m[0] and isinstance(m[1], nn.Conv2d):
                for p in m[1].parameters():
                    yield p
        elif key == "10x":
            if "backbone" not in m[0] and isinstance(m[1], nn.Conv2d):
                for p in m[1].parameters():
                    yield p

def log_confusion_mat(logger, confusion_mat, figsize, title, fmt, epoch):
    """
    Log confusion matrix to tensorboard as matplotlib figure.
    """
    fig = plt.figure(figsize=figsize)
    sn.heatmap(pd.DataFrame(confusion_mat), annot=True, fmt=fmt)
    plt.ylabel("True")
    plt.xlabel("Predicted")
    confusion_img = figure_to_image(fig, close=True)
    logger.add_image(title, confusion_img, epoch)

def normalise_confusion_mat(confusion_mat):
    normalised = np.zeros(confusion_mat.shape)

    for c_i in range(len(confusion_mat)):
        normalised[c_i] = confusion_mat[c_i] / confusion_mat[c_i].sum()

    return normalised

def postprocess(segs, dists, energy_cut, min_area):
    """
    Extract object instances from neural network outputs (seg and dist).
    Current pipeline:
        Binarise dist image at chosen energy level (lower=black, higher=white);
        Find contours on dist image;
        Discard contours with small area;
        Calculate overlap of each contour with seg image and assign class to blobs;
        Discard contours whose class is background;
        Grow remaining contours (How? e.g. reintegrating lower level into them).

    Args:
        - segs (4D float ndarray).
        - dists (4D float ndarray).
        - energy_cut (int): energy level to binarize image at.
        - min_area (int): minimum area of a contour in pixels.
    """

    pp_segs = segs.cpu().byte().numpy()
    print(pp_segs.shape)
    pp_dists = torch.argmax(dists, dim=1).cpu().byte().numpy()

    for seg, dist in zip(pp_segs, pp_dists):

        # This block is super fast (0.5 ms)
        _, thres = cv2.threshold(np.copy(dist), energy_cut, 255, cv2.THRESH_BINARY)
        _, contours, _ = cv2.findContours(np.copy(thres), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # This list comprehension is quite expensive (8 ms)
        contours = [c for c in contours if cv2.contourArea(c) >= min_area*5]

        instances = [0] * len(contours)

        for c, contour in enumerate(contours):

            for seg_class in seg:

                # Create binary mask
                mask = np.zeros((dist.shape[0], dist.shape[1]), dtype="uint8")
                cv2.drawContours(mask, [contour], -1, 255, -1)

                # Mask each class "hotmap" 
                seg_class = cv2.bitwise_and(seg_class, seg_class, mask=mask)

                cv2.imshow("img", seg)
                cv2.waitKey(0)
            
            instances[c] = seg   #TODO: convert to a dictionary with class, masks and bbox

    return instances


def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation="nearest")

