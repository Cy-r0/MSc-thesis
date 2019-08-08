from datetime import datetime
from itertools import chain
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

from datasets.voc_distance import VOCDistance
from datasets.voc_dual_task import VOCDualTask

from models.deeplabv3plus_multitask import Deeplabv3plus_multitask
from models.sync_batchnorm import DataParallelWithCallback
from models.sync_batchnorm.replicate import patch_replication_callback
import transforms.transforms as myT
from settings import VOCSettings
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
                       myT.Resize(sett.DATA_RESCALE),
                       myT.RandomCrop(sett.DATA_RANDOMCROP),
                       myT.ToTensor(),
                       #myT.Normalise((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ])

# Dataset setup
VOCW_train = VOCDualTask(sett.DSET_ROOT,
                         image_set="train_reduced",
                         transform=transform)
VOCW_val = VOCDualTask(sett.DSET_ROOT,
                       image_set="val_reduced",
                       transform=transform)

loader_train = DataLoader(VOCW_train,
                          batch_size = sett.TRAIN_BATCH_SIZE,
                          sampler=sampler.SubsetRandomSampler(
                              range(int(len(VOCW_train) * (1 - sett.VAL_FRACTION)))),
                          num_workers = sett.DATALOADER_JOBS,
                          pin_memory = True,
                          drop_last=True)
loader_val = DataLoader(VOCW_train,
                        batch_size = sett.VAL_BATCH_SIZE,
                        sampler=sampler.SubsetRandomSampler(
                            range(int(len(VOCW_train) * (1 - sett.VAL_FRACTION)),
                                  len(VOCW_train))),
                        num_workers = sett.DATALOADER_JOBS,
                        pin_memory = True,
                        drop_last=True)
loader_test = DataLoader(VOCW_val,
)


# Initialise tensorboardX logger
now = datetime.now()
if sett.LOG:
    tbX_logger = SummaryWriter(os.path.join(sett.LOG_PATH,
                                            now.strftime("%Y%m%d-%H%M")))

# Model setup
model = Deeplabv3plus_multitask(seg_classes=sett.N_CLASSES,
                                dist_classes=sett.N_ENERGY_LEVELS)

# Move model to GPU devices
device = torch.device(0)
print("GPUs found:", torch.cuda.device_count(), "\tGPUs used:", sett.TRAIN_GPUS)

if sett.TRAIN_GPUS > 1:
    model = nn.DataParallel(model)

# Load pretrained model weights only for backbone network
if sett.RESUME:
    current_dict = model.state_dict()
    pretrained_dict = torch.load(os.path.join(sett.PRETRAINED_PATH,
                        "deeplabv3plus_xception_VOC2012_epoch46_all.pth"))
    pretrained_dict = {k: v for k, v in pretrained_dict.items()
                       if "backbone" in k}

    current_dict.update(pretrained_dict)
    model.load_state_dict(current_dict)

model.to(device)

# Set losses (first one for semantics, second one for watershed)
seg_criterion = nn.CrossEntropyLoss(ignore_index=255)
dist_criterion = nn.CrossEntropyLoss(weight=torch.tensor(261500).div(torch.tensor(
                                                        [14000, 25500,
                                                         23000, 25000,
                                                         24000, 22500,
                                                         22500, 22000,
                                                         24000, 58500],
                                                         dtype=torch.float))
                                                        .to(device))

optimiser = optim.SGD(
        params = [
            {'params': helpers.get_params(model, key='1x'), 'lr': sett.TRAIN_LR},
            {'params': helpers.get_params(model, key='10x'), 'lr': 10 * sett.TRAIN_LR}
        ],
        momentum=sett.TRAIN_MOMENTUM)

# counts = [0] * sett.N_ENERGY_LEVELS

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)



# Training loop
i = 0
max_i = sett.TRAIN_EPOCHS * len(loader_train)
for epoch in range(sett.TRAIN_EPOCHS):

    train_loss = 0.0
    train_seg_loss = 0.0
    train_dist_loss = 0.0
    train_seg_acc = 0.0
    train_dist_acc = 0.0

    # Only initialise confusion matrices when they're going to be logged
    if epoch % sett.IMG_LOG_EPOCHS == 0:
        train_seg_confusion = np.zeros((sett.N_CLASSES + 1, sett.N_CLASSES + 1), dtype="float")
        train_dist_confusion = np.zeros((sett.N_ENERGY_LEVELS, sett.N_ENERGY_LEVELS), dtype="float")
        val_seg_confusion = np.zeros((sett.N_CLASSES + 1, sett.N_CLASSES + 1), dtype="float")
        val_dist_confusion = np.zeros((sett.N_ENERGY_LEVELS, sett.N_ENERGY_LEVELS), dtype="float")
    
    # Initialise tqdm
    tqdm_loader_train = tqdm(loader_train, ascii=True, desc="Train")

    model.train()
    for train_batch_i, train_batch in enumerate(tqdm_loader_train):

        train_inputs = train_batch["image"].to(device)
        # Labels need to be converted from float 0-1 to integers
        train_seg = train_batch["seg"].mul(255).round().long().squeeze(1).to(device)
        train_dist = train_batch["dist"].mul(255).round().long().squeeze(1).to(device)

        if sett.ADJUST_LR:
            lr = helpers.adjust_lr(optimiser, i, max_i, sett.TRAIN_LR, sett.TRAIN_POWER)
        else:
            lr = sett.TRAIN_LR

        optimiser.zero_grad()

        train_predicted_seg, train_predicted_dist = model(train_inputs)

        if epoch == 0:
            # Convert images to uint8 before postprocessing
            pp_seg = train_predicted_seg.cpu().byte().numpy()
            pp_dist = torch.argmax(train_predicted_dist, dim=1).cpu().byte().numpy()

            helpers.postprocess(train_predicted_seg, train_predicted_dist, energy_cut=1, min_area=int(sett.DATA_RESCALE ** 2 * 0.001))

        # Calculate losses
        seg_loss = seg_criterion(train_predicted_seg, train_seg)
        dist_loss = dist_criterion(train_predicted_dist, train_dist)
        loss = seg_loss + dist_loss
        loss.backward()
        optimiser.step()

        train_loss += loss.item()
        train_seg_loss += seg_loss.item()
        train_dist_loss += dist_loss.item()

        i += 1

        # Calculate accuracies
        train_seg_acc += (torch.sum(train_seg == torch.argmax(train_predicted_seg, dim=1))
                          .float().div(sett.DATA_RESCALE ** 2 * sett.TRAIN_BATCH_SIZE)).data
        train_dist_acc += (torch.sum(train_dist == torch.argmax(train_predicted_dist, dim=1))
                           .float().div(sett.DATA_RESCALE ** 2 * sett.TRAIN_BATCH_SIZE)).data

        # Only calculate confusion matrices every few epochs
        if epoch % sett.IMG_LOG_EPOCHS == 0:

            total_seg_labels = train_seg.cpu().flatten()
            total_seg_predictions = torch.argmax(train_predicted_seg, dim=1).cpu().flatten()
            train_seg_confusion += confusion_matrix(total_seg_labels, total_seg_predictions,
                                                    labels=list(chain(range(sett.N_CLASSES), [255])))
            train_seg_confusion /= sett.TRAIN_BATCH_SIZE

            total_dist_labels = train_dist.cpu().flatten()
            total_dist_predictions = torch.argmax(train_predicted_dist, dim=1).cpu().flatten()
            train_dist_confusion += confusion_matrix(total_dist_labels, total_dist_predictions,
                                                        labels=list(range(sett.N_ENERGY_LEVELS)))
            train_dist_confusion /= sett.TRAIN_BATCH_SIZE

        # Accumulate pixel belonging to each class (for weighted loss)
        #for class_i in range(sett.ENERGY_LEVELS):
         #   counts[class_i] += torch.nonzero(train_labels.flatten() == class_i).flatten().size(0)

    # Print counts
    #counts = [c / (len(loader_train) * sett.TRAIN_BATCH_SIZE) for c in counts]
    #print("Class counts per image:")
    #pprint(counts)


    val_loss = 0.0
    val_seg_loss = 0.0
    val_dist_loss = 0.0
    val_seg_acc = 0.0
    val_dist_acc = 0.0

    # Initialise tqdm
    tqdm_loader_val = tqdm(loader_val, ascii=True, desc="Valid")

    model.eval()
    with torch.no_grad():
        for val_batch_i, val_batch in enumerate(tqdm_loader_val):

            val_inputs = val_batch["image"].to(device)
            val_seg = val_batch["seg"].mul(255).round().long().squeeze(1).to(device)
            val_dist = val_batch["dist"].mul(255).round().long().squeeze(1).to(device)

            #start.record()
            val_predicted_seg, val_predicted_dist = model(val_inputs)
            #end.record()
            #torch.cuda.synchronize()
            #print("forward time:", start.elapsed_time(end))

            # Calculate losses
            seg_loss = seg_criterion(val_predicted_seg, val_seg)
            dist_loss = dist_criterion(val_predicted_dist, val_dist)
            loss = seg_loss + dist_loss
            val_loss += loss.item()
            val_seg_loss += seg_loss.item()
            val_dist_loss += dist_loss.item()

            # Calculate accuracies
            val_seg_acc += (torch.sum(val_seg == torch.argmax(val_predicted_seg, dim=1))
                            .float().div(sett.DATA_RESCALE ** 2 * sett.TRAIN_BATCH_SIZE)).data
            val_dist_acc += (torch.sum(val_dist == torch.argmax(val_predicted_dist, dim=1))
                             .float().div(sett.DATA_RESCALE ** 2 * sett.TRAIN_BATCH_SIZE)).data

            if epoch % sett.IMG_LOG_EPOCHS == 0:
                # Accumulate labels and predictions for confusion matrices
                total_seg_labels = val_seg.cpu().flatten()
                total_seg_predictions = torch.argmax(val_predicted_seg, dim=1).cpu().flatten()
                val_seg_confusion += confusion_matrix(total_seg_labels, total_seg_predictions,
                                                        labels=list(chain(range(sett.N_CLASSES), [255])))
                val_seg_confusion /= sett.VAL_BATCH_SIZE

                total_dist_labels = val_dist.cpu().flatten()
                total_dist_predictions = torch.argmax(val_predicted_dist, dim=1).cpu().flatten()
                val_dist_confusion += confusion_matrix(total_dist_labels, total_dist_predictions,
                                                        labels=list(range(sett.N_ENERGY_LEVELS)))
                val_dist_confusion /= sett.VAL_BATCH_SIZE

    # Average losses on all batches
    train_loss /= len(loader_train)
    train_seg_loss /= len(loader_train)
    train_dist_loss /= len(loader_train)
    train_seg_acc /= len(loader_train)
    train_dist_acc /= len(loader_train)

    # Average accuracies on batches
    val_loss /= len(loader_val)
    val_seg_loss /= len(loader_val)
    val_dist_loss /= len(loader_val)
    val_seg_acc /= len(loader_val)
    val_dist_acc /= len(loader_val)


    print("Epoch: %d/%d\ti: %d\tlr: %g\ttrain_loss: %g\tval_loss: %g\n"
          % (epoch+1, sett.TRAIN_EPOCHS, i, lr, train_loss, val_loss))

    # Convert training data for plotting
    train_input_tb = make_grid(train_inputs).cpu().numpy()
    train_seg_tb = make_grid(helpers.colormap(train_seg.float().div(sett.N_CLASSES)
                                        .unsqueeze(1).cpu())).numpy()
    train_seg_prediction_tb = make_grid(helpers.colormap(torch.argmax(train_predicted_seg, dim=1)
                                              .float().div(sett.N_CLASSES).unsqueeze(1)
                                              .cpu())).numpy()
    train_dist_tb = make_grid(helpers.colormap(train_dist.float().div(sett.N_ENERGY_LEVELS)
                                        .unsqueeze(1).cpu())).numpy()
    train_dist_prediction_tb = make_grid(helpers.colormap(torch.argmax(train_predicted_dist, dim=1)
                                              .float().div(sett.N_ENERGY_LEVELS).unsqueeze(1)
                                              .cpu())).numpy()                                         

    # Convert val data for plotting
    val_input_tb = make_grid(val_inputs).cpu().numpy()
    val_seg_tb = make_grid(helpers.colormap(val_seg.float().div(sett.N_CLASSES)
                                      .unsqueeze(1).cpu())).numpy()
    val_seg_prediction_tb = make_grid(helpers.colormap(torch.argmax(val_predicted_seg, dim=1)
                                            .float().div(sett.N_CLASSES).unsqueeze(1)
                                            .cpu())).numpy()
    val_dist_tb = make_grid(helpers.colormap(val_dist.float().div(sett.N_ENERGY_LEVELS)
                                        .unsqueeze(1).cpu())).numpy()
    val_dist_prediction_tb = make_grid(helpers.colormap(torch.argmax(val_predicted_dist, dim=1)
                                              .float().div(sett.N_ENERGY_LEVELS).unsqueeze(1)
                                              .cpu())).numpy()
    
    # Log scalars to tensorboardX
    tbX_logger.add_scalars("total_losses", {"train_loss": train_loss,
                                            "val_loss": val_loss}, epoch)
    tbX_logger.add_scalars("seg_losses", {"train_seg_loss": train_seg_loss,
                                          "val_seg_loss": val_seg_loss}, epoch)
    tbX_logger.add_scalars("dist_losses", {"train_dist_loss": train_dist_loss,
                                           "val_dist_loss": val_dist_loss}, epoch)
    tbX_logger.add_scalars("seg_acc", {"train_seg_acc": train_seg_acc,
                                       "val_seg_acc": val_seg_acc}, epoch)
    tbX_logger.add_scalars("dist_acc", {"train_dist_acc": train_dist_acc,
                                        "val_dist_acc": val_dist_acc}, epoch)                                 
    tbX_logger.add_scalar("lr", lr, epoch)

    # it seems like tensorboard doesn't like saving a lot of images,
    # so log images only once every few epochs
    if epoch % sett.IMG_LOG_EPOCHS == 0:

        # Training images
        tbX_logger.add_image("train_input", train_input_tb, epoch)
        tbX_logger.add_image("train_seg", train_seg_tb, epoch)
        tbX_logger.add_image("train_seg_prediction", train_seg_prediction_tb, epoch)
        tbX_logger.add_image("train_dist", train_dist_tb, epoch)    
        tbX_logger.add_image("train_dist_prediction", train_dist_prediction_tb, epoch)
        # Validation images
        tbX_logger.add_image("val_input", val_input_tb, epoch)
        tbX_logger.add_image("val_seg", val_seg_tb, epoch)
        tbX_logger.add_image("val_seg_prediction", val_seg_prediction_tb, epoch)
        tbX_logger.add_image("val_dist", val_dist_tb, epoch)    
        tbX_logger.add_image("val_dist_prediction", val_dist_prediction_tb, epoch)

        # Divide unnormalised matrices by n. of image pixels
        train_seg_confusion /= len(loader_train)
        train_dist_confusion /= len(loader_train)
        val_seg_confusion /= len(loader_val)
        val_dist_confusion /= len(loader_val)

        # Normalise confusion matrices
        train_seg_confusion_n = helpers.normalise_confusion_mat(train_seg_confusion)
        train_dist_confusion_n = helpers.normalise_confusion_mat(train_dist_confusion)
        val_seg_confusion_n = helpers.normalise_confusion_mat(val_seg_confusion)
        val_dist_confusion_n = helpers.normalise_confusion_mat(val_dist_confusion)

        # Log confusion matrices to tensorboard
        helpers.log_confusion_mat(tbX_logger, train_seg_confusion, (16,10), "train_confusion_seg", "0.0f", epoch)
        helpers.log_confusion_mat(tbX_logger, train_dist_confusion, (9,7), "train_confusion_dist", "0.0f", epoch)
        helpers.log_confusion_mat(tbX_logger, val_seg_confusion, (16,10), "val_confusion_seg", "0.0f", epoch)
        helpers.log_confusion_mat(tbX_logger, val_dist_confusion, (9,7), "val_confusion_dist", "0.0f", epoch)

        # Log normalised confusion matrices to tensorboard
        helpers.log_confusion_mat(tbX_logger, train_seg_confusion_n, (16,10), "train_confusion_seg_n", "0.3f", epoch)
        helpers.log_confusion_mat(tbX_logger, train_dist_confusion_n, (9,7), "train_confusion_dist_n", "0.3f", epoch)
        helpers.log_confusion_mat(tbX_logger, val_seg_confusion_n, (16,10), "val_confusion_seg_n", "0.3f", epoch)
        helpers.log_confusion_mat(tbX_logger, val_dist_confusion_n, (9,7), "val_confusion_dist_n", "0.3f", epoch)        


    # Save checkpoint
    if epoch % 100 == 0 and epoch != 0:
        save_path = os.path.join(sett.PRETRAINED_PATH, "%s_%s_%s_epoch%d.pth"
                    %(sett.MODEL_NAME, sett.MODEL_BACKBONE, sett.DATA_NAME, epoch))
        torch.save(model.state_dict(), save_path)
        print("%s has been saved." %save_path)

tbX_logger.close()

# Save final trained model
save_path = os.path.join(sett.PRETRAINED_PATH, "%s_%s_%s_epoch%d_final.pth"
            %(sett.MODEL_NAME, sett.MODEL_BACKBONE, sett.DATA_NAME, sett.TRAIN_EPOCHS))
torch.save(model.state_dict(), save_path)
print("FINISHED: %s has been saved." %save_path)

# list of TODO's:

# find out what layers[0] means for the backbone network

# try focal loss
# implement two-head model
# add intermediate vector stage (maybe not needed)
# implement coco dataset training