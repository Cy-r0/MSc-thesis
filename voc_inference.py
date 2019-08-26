from datetime import datetime
from itertools import chain
import json
import os
from pprint import pprint
from timeit import default_timer

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.metrics import confusion_matrix
import timeit
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
import torch.optim as optim
from torch.utils.data import DataLoader, sampler
import torchvision.transforms as T
from torchvision.utils import make_grid
from tqdm import tqdm
from pycocotools import mask

from datasets.voc_dual_task import VOCDualTask
from models.deeplabv3plus_multitask import Deeplabv3plus_multitask
import transforms.transforms as myT
from config.config import VOCConfig
import utils.utils as utils


cfg = VOCConfig()

# Fix all seeds for reproducibility
seed = 777
np.random.seed(seed)
torch.manual_seed(seed)


def setup_dataloader():
    """
    Initialise dataset, sampler and return dataloader.
    """

    # Dataset transforms
    transform = T.Compose([
        myT.Quantise(level_widths=cfg.LEVEL_WIDTHS),
        #myT.Resize(cfg.DATA_RESCALE),
        myT.ToTensor()])

    # Datasets
    VOC_val = VOCDualTask(
        cfg.DSET_ROOT,
        image_set="val",
        transform=transform)

    # Data loaders
    loader_val = DataLoader(
        VOC_val,
        batch_size=cfg.TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.DATALOADER_JOBS,
        pin_memory=True,
        drop_last=False)
    
    return loader_val


def setup_model(device):
    """
    Initialise model and load weights.
    """
    
    # Initialise model
    model = Deeplabv3plus_multitask(
        seg_classes=cfg.N_CLASSES,
        dist_classes=cfg.N_ENERGY_LEVELS)
    model.to(device)
    print(
        "GPUs found:", torch.cuda.device_count(),
        "\tGPUs used:", cfg.TEST_GPUS)

    # Load trained weights TODO: fix loading
    current_dict = model.state_dict()
    pretrained_dict = torch.load(
        os.path.join(
            cfg.PRETRAINED_PATH,
            "deeplabv3plus_multitask_xception_VOC2012_epoch40_final_highweight.pth"))

    # Get rid of the word "module" in keys,
    # since this model is not being used with dataparallel anymore
    pretrained_dict = { 
        k[7:]: v for k, v in pretrained_dict.items()
    }

    current_dict.update(pretrained_dict)
    model.load_state_dict(current_dict)

    return model


def inference():

    # Check that GPUs and batch size are 1
    assert cfg.TEST_GPUS == 1 and cfg.TEST_BATCH_SIZE == 1, \
        "This code only runs on one GPU and with a batch size of 1"

    # Dataloaders
    loader_val = setup_dataloader()

    # Model setup
    device = torch.device(0)
    model = setup_model(device)
        
    # Initialise tqdm
    loader_val = tqdm(loader_val, ascii=True)

    json_data = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    model.eval()
    timing_avg = []

    with torch.no_grad():

        softmax = nn.Softmax2d()

        for batch in loader_val:

            img_id = int(batch["img_id"][0])
            inputs = batch["image"].to(device)
            seg = batch["seg"]
            dist = batch["dist"]

            # calculate size to rescale to
            original_h = inputs.shape[2]
            original_w = inputs.shape[3]
            if original_h > original_w:
                rescaled_size = (
                    cfg.DATA_RESCALE * original_h // original_w,
                    cfg.DATA_RESCALE
                )
            else:
                rescaled_size = (
                    cfg.DATA_RESCALE,
                    cfg.DATA_RESCALE * original_w // original_h
                )

            # Resize images
            inputs = F.interpolate(
                inputs,
                size=rescaled_size,
                mode="bilinear")
            seg = F.interpolate(
                seg,
                size=rescaled_size,
                mode="nearest")
            dist = F.interpolate(
                dist,
                size=rescaled_size,
                mode="nearest")

            # Labels need to be converted from float 0-1 to integers
            seg = batch["seg"].mul(255).round().long().squeeze(1).to(device)
            dist = batch["dist"].mul(255).round().long().squeeze(1).to(device)
        
            seg = torch.unsqueeze(seg, 0)
            dist = torch.unsqueeze(dist, 0)

            start.record()

            predicted_seg, predicted_dist = model(inputs)

            end.record()
            torch.cuda.synchronize()
            print("model time", start.elapsed_time(end))

            # rescale predictions to original size
            predicted_seg = F.interpolate(
                predicted_seg,
                size=(original_h, original_w),
                mode="nearest")
            predicted_dist = F.interpolate(
                predicted_dist,
                size=(original_h, original_w),
                mode="nearest")

            # Softmax predictions
            predicted_seg = softmax(predicted_seg)
            predicted_dist = softmax(predicted_dist)
            
            use_gndtruth = False

            if use_gndtruth:
                for pred_seg, pred_dist in zip(seg, dist):

                    pred_seg = pred_seg.cpu()
                    pred_seg[pred_seg == 255] = 0
                    pred_seg_onehot = torch.FloatTensor(
                        22,
                        pred_seg.shape[1],
                        pred_seg.shape[2]) \
                        .zero_()
                    pred_seg_onehot = pred_seg_onehot.scatter(0, pred_seg, 1)

                    tic = timeit.default_timer()

                    instances = utils.postprocess(pred_seg_onehot, pred_dist, energy_cut=0)

                    toc = timeit.default_timer()
                    print("TOTAL postprocessing time: %.2f" %((toc-tic)*1000))

                    if instances:
                        for instance in instances:
                            # encoded bytestring in mask needs to be converted to ascii
                            encoded_mask = mask.encode(np.asfortranarray(instance["mask"]))
                            encoded_mask["counts"] = encoded_mask["counts"].decode("ascii")
                            instance_dict = {
                                "image_id": img_id,
                                "category_id": int(np.argmax(instance["scores"])),
                                "segmentation": encoded_mask,
                                "score": max(instance["scores"])
                            }
                            json_data.append(instance_dict)

            else:
                # use actual predictions from neural network
                for pred_seg, pred_dist in zip(predicted_seg, predicted_dist):

                    tic = timeit.default_timer()
                    
                    torch.cuda.synchronize()
                    instances = utils.postprocess_2(pred_seg, pred_dist, energy_cut=1)

                    torch.cuda.synchronize()
                    toc = timeit.default_timer()
                    timing_avg.append((toc-tic) * 1000)
                    print("Cumulative avg pp time: %.2f" %(sum(timing_avg)/len(timing_avg)))


                    if instances:
                        for instance in instances:

                            # encoded bytestring in mask needs to be converted to ascii
                            encoded_mask = mask.encode(np.asfortranarray(instance["segmentation"]))
                            encoded_mask["counts"] = encoded_mask["counts"].decode("ascii")
                            instance["segmentation"] = encoded_mask

                            # Add image id to instance dictionary
                            instance["image_id"] = img_id

                            json_data.append(instance)


    # Write results to json
    with open("COCO_style_results/voc2012_val_deeplab_highweight.json", 'w') as f:
        json.dump(json_data, f)


if __name__ == "__main__":
    
    inference()