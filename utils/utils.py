import warnings

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


def log_confusion_mat(logger, confusion_mat, figsize, title, fmt, epoch, xlabel, ylabel):
    """
    Log confusion matrix to tensorboard as matplotlib figure.
    """
    fig = plt.figure(figsize=figsize)
    sn.heatmap(pd.DataFrame(confusion_mat), annot=True, fmt=fmt, xticklabels=xlabel,
                                                                 yticklabels=ylabel)
    plt.ylabel("True")
    plt.xlabel("Predicted")
    confusion_img = figure_to_image(fig, close=True)
    logger.add_image(title, confusion_img, epoch)


def normalise_confusion_mat(confusion_mat):

    # Initialise container for normalised matrix
    normalised = np.zeros(confusion_mat.shape)

    for c_i in range(len(confusion_mat)):
        # Avoid division by zero
        if confusion_mat[c_i].sum() != 0:
            normalised[c_i] = confusion_mat[c_i] / confusion_mat[c_i].sum()
        else:
            warnings.warn("Row of confusion matrix is zero")
            normalised[c_i] = confusion_mat[c_i]

    return normalised


def postprocess(seg, dist, energy_cut, min_area=40, debug=True):
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
        - seg (3D float ndarray).
        - dist (3D float ndarray).
        - energy_cut (int): energy level to binarize image at.
        - min_area (int): minimum area of a contour in pixels (empirically determined).
        - debug (bool): boolean flag that shows how images are processed.
    """

    seg = seg.cpu().detach().numpy()
    #seg = torch.clamp(seg, 0.001, 1).cpu().detach().numpy()
    
    dist = torch.argmax(dist, dim=0).cpu().byte().numpy()
    #dist = dist.squeeze(dim=0).cpu().byte().numpy()

    instances = []

    # Show distance img
    if debug:
        plt.imshow(dist / 255)
        plt.show()

    # Cut image at energy level
    _, thres = cv2.threshold(np.copy(dist), energy_cut, 255, cv2.THRESH_BINARY)
    _, contours, hierarchy = cv2.findContours(
        np.copy(thres), 
        cv2.RETR_TREE, 
        cv2.CHAIN_APPROX_NONE)

    # Get rid of small contours (they often appear due to noise)
    #contours = [c for c in contours if cv2.contourArea(c) >= min_area]

    # Show all contours
    if debug:
        distcopy = np.copy(dist)
        cv2.drawContours(distcopy, contours, -1, 255, 1)
        cv2.imshow("all contours bigger than min_area", distcopy)
        cv2.waitKey(0)

    for c, contour in enumerate(contours):

        # Create binary mask of contoured region
        mask = np.zeros((dist.shape[0], dist.shape[1]), dtype="uint8")
        cv2.drawContours(mask, [contour], -1, 255, -1)

        # Create binary mask for children
        children_mask = np.zeros((dist.shape[0], dist.shape[1]), dtype="uint8")

        # Find first child
        child_idx = hierarchy[0, c, 2]

        # Overlay masks of all children
        while child_idx != -1:
            cv2.drawContours(children_mask, [contours[child_idx]], -1, 255, -1)
            #print("child is", child_idx)
            #cv2.imshow("children mask", children_mask)
            #cv2.waitKey(0)

            child_idx = hierarchy[0, child_idx, 0]

        # Subtract children mask from parent mask to preserve holes
        mask = cv2.bitwise_xor(mask, children_mask)

        scores = [0] * len(seg)

        for s, seg_class in enumerate(seg):

            # Get rid of semantic pixels outside the mask
            seg_class = cv2.bitwise_and(seg_class, seg_class, mask=mask)

            # average pixel values for object classification
            area = cv2.countNonZero(seg_class)
            average = cv2.sumElems(seg_class)[0] / area
            assert average >= 0 and average <= 1
            scores[s] = average
        
        # Check if contour contains only low energy levels
        # You need to get rid of the perimeter of the mask
        perimeter = np.zeros((dist.shape[0], dist.shape[1]), dtype="uint8")
        cv2.drawContours(perimeter, [contour], -1, 255, 1)
        mask_without_perimeter = cv2.bitwise_xor(perimeter, mask)
        # Mask distance with the mask without perimeter
        masked_dist = cv2.bitwise_and(dist, dist, mask=mask_without_perimeter)

        # Find which energy levels are inside the contour
        energy_inside = np.unique(masked_dist)
        contains_only_lower_levels = all(l <= energy_cut for l in energy_inside)
        print(energy_inside)
        print(contains_only_lower_levels)

        # Discard background, unlabelled and small instances
        if (np.argmax(scores) != 0
            and np.argmax(scores) != 21
            and area > min_area
            and not contains_only_lower_levels):
            instance_dict = {
                "mask": mask,
                "scores": scores
            }
            instances.append(instance_dict)

            # Show final mask and print class
            if debug:
                print("class:", np.argmax(scores))
                cv2.imshow("mask", mask)
                cv2.waitKey(0)

    # Return a list of instances for each image in the batch
    return instances


def show(img):
    npimg = img.cpu().numpy()

    if len(img.shape) == 3:
        plt.imshow(np.transpose(npimg, (1,2,0)), interpolation="nearest")
    elif len(img.shape) == 2:
        plt.imshow(npimg, interpolation="nearest")

