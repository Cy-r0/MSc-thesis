import warnings
import timeit

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from tensorboardX.utils import figure_to_image
import torch
import torch.nn as nn



def adjust_lr(optimiser, i, max_i, initial_lr, pow):
    """
    Gradually decrease learning rate as iterations increase.
    """
    lr = initial_lr * (1 - i/(max_i + 1)) ** pow
    for param_group in optimiser.param_groups:
        param_group["lr"] = lr
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


def postprocess_old(seg, dist, energy_cut, min_area=80, debug=False):
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

    #tic = timeit.default_timer()


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
    #PERF: findContours is the most expensive function (3-20 ms)
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

    #toc = timeit.default_timer()
    #print("find contours time: %.2f" %((toc-tic)*1000))


    for c, contour in enumerate(contours):

        # Filter out smaller contours
        if cv2.contourArea(contour) >= min_area:


            #tic = timeit.default_timer()

            #PERF: this part up to next PERF takes about 0.4 ms

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


            #toc = timeit.default_timer()
            #print("mask creation time:", toc-tic)

            #tic = timeit.default_timer()


            # Calculate area of instance mask
            mask_area = cv2.countNonZero(mask)
            
            # Initialise scores list
            scores = [0] * seg.shape[0]

            for s, seg_class in enumerate(seg):

                # Get rid of semantic pixels outside the mask

                #tik = timeit.default_timer()

                seg_class = cv2.bitwise_and(seg_class, seg_class, mask=mask)

                #tok = timeit.default_timer()
                #print("masking timeL", tok-tik)

                # Calculate scores for each class by averaging pixel scores
                average = cv2.sumElems(seg_class)[0] / mask_area

                assert average >= 0 and average <= 1

                scores[s] = average
            
            #toc = timeit.default_timer()
            #print("semantic time:", toc-tic)


            #tic = timeit.default_timer()
            
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
            #print(energy_inside)
            #print(contains_only_lower_levels)

            # Discard background, unlabelled and low energy instances
            if (np.argmax(scores) != 0
                and np.argmax(scores) != 21
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
            
            #toc = timeit.default_timer()
            #print("low energy exclusion time: %.2f" %((toc-tic)*1000))

    # Return a list of instances for each image in the batch
    return instances


def show(img):
    npimg = img.cpu().numpy()

    if len(img.shape) == 3:
        plt.imshow(np.transpose(npimg, (1,2,0)), interpolation="nearest")
    elif len(img.shape) == 2:
        plt.imshow(npimg, interpolation="nearest")


def postprocess(seg, dist, energy_cut, min_area=900, check_lowenergy=False, debug=True):
    """
    Extract object instances from neural network outputs (seg and dist).

    Args:
        - seg (3D float ndarray).
        - dist (3D float ndarray).
        - energy_cut (int): energy level to binarize image at.
        - min_area (int): minimum area of a contour in pixels (empirically determined).
        - debug (bool): boolean flag that shows how images are processed.
    """

    seg_argmax = torch.argmax(seg, dim=0).long() # TODO: check if byte is faster
    dist = torch.argmax(dist, dim=0).long()

    # Show distance img
    if debug:
        dist_np = dist.cpu().numpy()
        plt.imshow(dist_np / 255)
        plt.show()
        seg_np = seg_argmax.cpu().numpy()
        plt.imshow(seg_np / 255)
        plt.show()

    # Threshold dist at chosen energy level
    dist_thres = torch.where(
        dist > energy_cut,
        torch.tensor(1).cuda(),
        torch.tensor(0).cuda())

    # Show thresholded dist
    if debug:
        thres_np = dist_thres.cpu().numpy()
        plt.imshow(thres_np)
        plt.show()

    # Mask semantic with thresholded dist
    masked_seg = torch.where(
        dist_thres > 0,
        seg_argmax,
        dist_thres).byte()

    # Show masked seg
    if debug:
        seg_np = masked_seg.cpu().numpy()
        plt.imshow(seg_np)
        plt.show()

    # Binarise segmentation and convert to numpy before feeding to findContours()
    # NOTE: opencv only accepts uint8 as a format
    binarised_seg = torch.where(
        masked_seg > 0,
        torch.tensor(1).cuda(),
        torch.tensor(0).cuda()).byte().cpu().numpy()

    # Find contours. PERF: findContours is the most expensive function (3-20 ms)
    _, contours, hierarchy = cv2.findContours(
        binarised_seg,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_NONE)
    
    # Show thresholded segmentation
    if debug:
        seg_copy = np.copy(binarised_seg)
        cv2.drawContours(seg_copy, contours, -1, 255, 1)
        plt.imshow(seg_copy)
        plt.show()
 
    instances = []

    # Iterate through contours
    for c, contour in enumerate(contours):

        if cv2.contourArea(contour) >= min_area:

            # Create binary mask of contour
            contour_mask = np.zeros((dist.shape[0], dist.shape[1]), dtype="uint8")
            cv2.drawContours(contour_mask, [contour], -1, 255, -1)
            contour_mask_area = cv2.countNonZero(contour_mask)

            # Create binary mask for children contours
            children_mask = np.zeros((dist.shape[0], dist.shape[1]), dtype="uint8")

            # Find first child
            child_idx = hierarchy[0, c, 2]

            # Overlay masks of all children
            while child_idx != -1:
                cv2.drawContours(children_mask, [contours[child_idx]], -1, 255, -1)
                child_idx = hierarchy[0, child_idx, 0]
                
            # Subtract children mask from parent mask to preserve holes
            contour_mask = cv2.bitwise_xor(contour_mask, children_mask)

            if debug:
                cv2.imshow("c mask", contour_mask)
                cv2.waitKey(0)

            
            # Check if contour contains only low energy levels
            if check_lowenergy:
                # You need to get rid of the perimeter of the mask
                perimeter = np.zeros((dist.shape[0], dist.shape[1]), dtype="uint8")
                cv2.drawContours(perimeter, [contour], -1, 255, 1)
                mask_without_perimeter = cv2.bitwise_xor(perimeter, contour_mask)
                # Mask distance with the mask without perimeter
                dist_np = dist.cpu().numpy()
                masked_dist = cv2.bitwise_and(dist_np, dist_np, mask=mask_without_perimeter)

                # Find which energy levels are inside the contour
                energy_inside = np.unique(masked_dist)
                contains_only_lower_levels = all(l <= energy_cut for l in energy_inside)
            else:
                contains_only_lower_levels = False

            # Ignore contours that only contain low energy
            if not contains_only_lower_levels:

                # Dilate binary mask to recover lower energy levels
                dilation_k = np.ones((21,21), np.uint8)
                contour_mask = cv2.dilate(contour_mask, dilation_k, iterations=1)

                if debug:
                    cv2.imshow("c mask dilated", contour_mask)
                    cv2.waitKey(0)

                # Move binary mask to gpu and mask segmentation tensor
                contour_mask = torch.tensor(contour_mask).cuda()
                contour_seg = torch.where(
                    contour_mask > 0,
                    seg,
                    contour_mask.float())

                # Average all 2d maps in segmentation tensor to find scores
                scores = torch.sum(contour_seg, dim=(1,2)) / contour_mask_area

                # Append 
                instance_dict = {
                    "category_id": torch.argmax(scores).item(),
                    "segmentation": contour_mask.cpu().numpy(),
                    "score": torch.max(scores).item()
                }
                instances.append(instance_dict)
    
    return instances