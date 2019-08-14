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
from tensorboardX import SummaryWriter
import timeit
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
import torch.optim as optim
from torch.utils.data import DataLoader, sampler
import torchvision.transforms as T
from torchvision.utils import make_grid
from tqdm import tqdm
from pycocotools import mask

from datasets.voc_distance import VOCDistance
from datasets.voc_dual_task import VOCDualTask
from models.deeplabv3plus_multitask import Deeplabv3plus_multitask
from models.sync_batchnorm import DataParallelWithCallback
from models.sync_batchnorm.replicate import patch_replication_callback
import transforms.transforms as myT
from config.config import VOCSettings
import utils.helpers as helpers


sett = VOCSettings()


#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


# Fix all seeds for reproducibility
np.random.seed(777)
torch.manual_seed(777)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Dataset transforms
transform = T.Compose([myT.Quantise(level_widths=sett.LEVEL_WIDTHS),
                       myT.ToTensor()])

# Dataset setup
VOCW_val = VOCDualTask(sett.DSET_ROOT,
                       image_set="train_reduced",
                       transform=transform)

loader_test = DataLoader(VOCW_val,
                        batch_size = sett.TEST_BATCH_SIZE,
                        pin_memory = True,
                        drop_last=True)

# Model setup
model = Deeplabv3plus_multitask(seg_classes=sett.N_CLASSES,
                                dist_classes=sett.N_ENERGY_LEVELS)

# Move model to GPU devices
device = torch.device(0)
print("GPUs found:", torch.cuda.device_count(), "\tGPUs used:", sett.TEST_GPUS)

if sett.TEST_GPUS > 1:
    model = nn.DataParallel(model)

# Load pretrained model weights only for backbone network
current_dict = model.state_dict()
trained_dict = torch.load(os.path.join(sett.TRAINED_PATH, "%s_%s_%s_epoch%d_final.pth"
                    %(sett.MODEL_NAME, sett.MODEL_BACKBONE, sett.DATA_NAME, sett.TRAIN_EPOCHS)))
trained_dict = {k: v for k, v in trained_dict.items()
                    if "backbone" in k}
# Get rid of "module" in pretrained dict keywords if my model is not DataParallel
if sett.TEST_GPUS == 1:
    trained_dict = {k[7:]: v for k, v in trained_dict.items()}
current_dict.update(trained_dict)
model.load_state_dict(current_dict)

model.to(device)

# Initialise logger
now = datetime.now()
tbX_logger = SummaryWriter(os.path.join(sett.LOG_PATH, now.strftime("%Y%m%d-%H%M") + "_test"))

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)


# Test loop

seg_acc = 0.0
dist_acc = 0.0

seg_confusion = np.zeros((sett.N_CLASSES, sett.N_CLASSES), dtype="float")
dist_confusion = np.zeros((sett.N_ENERGY_LEVELS, sett.N_ENERGY_LEVELS), dtype="float")

json_data = []
    
# Initialise tqdm
tqdm_loader_test = tqdm(loader_test, ascii=True)

model.eval()
for batch_i, batch in enumerate(tqdm_loader_test):

    inputs = batch["image"].to(device)
    # Labels need to be converted from float 0-1 to integers
    seg = batch["seg"].mul(255).round().long().squeeze(1).to(device)
    dist = batch["dist"].mul(255).round().long().squeeze(1).to(device)
   
    seg = torch.unsqueeze(seg, 0)
    dist = torch.unsqueeze(dist, 0)

    predicted_seg, predicted_dist = model(inputs)
    # Softmax predictions TODO: modify
    SM = nn.Softmax2d()
    predicted_seg = SM(predicted_seg)
    predicted_dist = SM(predicted_dist)
    
    #TODO: change back to predictions
    for pred_seg, pred_dist in zip(seg, dist):


        pred_seg = pred_seg.cpu()
        pred_seg[pred_seg == 255] = 0
        pred_seg_onehot = torch.FloatTensor(22, pred_seg.shape[1], pred_seg.shape[2]).zero_()
        pred_seg_onehot = pred_seg_onehot.scatter(0, pred_seg, 1)
        instances = helpers.postprocess(pred_seg_onehot, pred_dist, energy_cut=0)
        for instance in instances:
            # encoded bytestring in mask needs to be converted to ascii
            encoded_mask = mask.encode(np.asfortranarray(instance["mask"]))
            encoded_mask["counts"] = encoded_mask["counts"].decode("ascii")
            instance_dict = {
                "image_id": batch_i,
                "category_id": int(np.argmax(instance["scores"])),
                "segmentation": encoded_mask,
                "score": max(instance["scores"])
            }
            json_data.append(instance_dict)
"""
    # Calculate accuracies
    seg_acc += (torch.sum(seg == torch.argmax(predicted_seg, dim=1))
                          .float().div(inputs.shape[2] * inputs.shape[3] * sett.TEST_BATCH_SIZE)).data
    dist_acc += (torch.sum(dist == torch.argmax(predicted_dist, dim=1))
                          .float().div(inputs.shape[2] * inputs.shape[3] * sett.TEST_BATCH_SIZE)).data

    total_seg_labels = seg.cpu().flatten()
    total_seg_predictions = torch.argmax(predicted_seg, dim=1).cpu().flatten()
    seg_confusion += confusion_matrix(total_seg_labels, total_seg_predictions,
                                            labels=list(sett.CLASSES.keys()))
    seg_confusion /= sett.TEST_BATCH_SIZE

    total_dist_labels = dist.cpu().flatten()
    total_dist_predictions = torch.argmax(predicted_dist, dim=1).cpu().flatten()
    dist_confusion += confusion_matrix(total_dist_labels, total_dist_predictions,
                                                labels=list(range(sett.N_ENERGY_LEVELS)))
    dist_confusion /= sett.TEST_BATCH_SIZE
"""
# Write results to json
with open("results/voc2012_train_reduced.json", 'w') as f:
	json.dump(json_data, f)

# Average accuracies on batches
seg_acc /= len(loader_test)
dist_acc /= len(loader_test)
                                       

# Convert data for plotting
input_tb = make_grid(inputs).cpu().numpy()
seg_tb = make_grid(helpers.colormap(seg.float().div(sett.N_CLASSES).unsqueeze(1).cpu())).numpy()
seg_prediction_tb = make_grid(helpers.colormap(torch.argmax(predicted_seg, dim=1)
                                        .float().div(sett.N_CLASSES).unsqueeze(1)
                                        .cpu())).numpy()
dist_tb = make_grid(helpers.colormap(dist.float().div(sett.N_ENERGY_LEVELS).unsqueeze(1).cpu())).numpy()
dist_prediction_tb = make_grid(helpers.colormap(torch.argmax(predicted_dist, dim=1)
                                            .float().div(sett.N_ENERGY_LEVELS).unsqueeze(1)
                                            .cpu())).numpy()


tbX_logger.add_image("input", input_tb, epoch)
tbX_logger.add_image("seg", seg_tb, epoch)
tbX_logger.add_image("seg_prediction", seg_prediction_tb, epoch)
tbX_logger.add_image("dist", dist_tb, epoch)    
tbX_logger.add_image("dist_prediction", dist_prediction_tb, epoch)

# Divide unnormalised matrices
seg_confusion /= len(loader_test)
dist_confusion /= len(loader_test)

# Normalise confusion matrices
seg_confusion_n = helpers.normalise_confusion_mat(seg_confusion)
dist_confusion_n = helpers.normalise_confusion_mat(dist_confusion)

# Log confusion matrices to tensorboard
helpers.log_confusion_mat(tbX_logger, seg_confusion, (16,10), "train_confusion_seg", "0.0f", epoch)
helpers.log_confusion_mat(tbX_logger, dist_confusion, (9,7), "train_confusion_dist", "0.0f", epoch)
        
# Log normalised confusion matrices to tensorboard
helpers.log_confusion_mat(tbX_logger, seg_confusion_n, (16,10), "train_confusion_seg_n", "0.3f", epoch)
helpers.log_confusion_mat(tbX_logger, dist_confusion_n, (9,7), "train_confusion_dist_n", "0.3f", epoch)

tbX_logger.close()