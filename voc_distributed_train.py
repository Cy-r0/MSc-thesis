from datetime import datetime
from itertools import chain
import os
from pprint import pprint
import random
from timeit import default_timer
import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.metrics import confusion_matrix
from tensorboardX import SummaryWriter
import timeit
import torch
import torch.distributed as distr
import torch.multiprocessing as multiproc
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
import torch.optim as optim
from torch.utils.data import DataLoader, sampler
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as T
from torchvision.utils import make_grid
from tqdm import tqdm

from datasets.voc_dual_task import VOCDualTask
from models.losses.losses import MeanSquaredAngularLoss
from models.deeplabv3plus_multitask import Deeplabv3plus_multitask
import transforms.transforms as myT
from config.config import VOCConfig
import utils.utils as utils


cfg = VOCConfig()

# Fix all seeds for reproducibility
seed = 777
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False



def calc_iou(confusion_matrix):
    """
    Calculates per-class IoUs from numpy confusion matrix.
    """
    class_iou = [0] * len(confusion_matrix)

    for i in range(len(confusion_matrix)):

        intersection = confusion_matrix[i,i]
        union = np.sum(confusion_matrix[i,:]) + np.sum(confusion_matrix[:,i]) - intersection
    
        if union == 0:
            warnings.warn("IoU calculation: union is zero!")
            class_iou[i] = 0
        else:
            class_iou[i] = intersection / union
    
    return class_iou


def calc_confusion_matrix(target, prediction, labels):
    """
    Computes confusion matrix. NOTE: predictions need to be already argmaxed
    """

    target = target.flatten()
    prediction = prediction.flatten()

    conf_matrix = confusion_matrix(target, prediction, labels=labels)
    
    return conf_matrix


def setup_dataloaders(rank):
    """
    Initialise datasets, samplers and return dataloaders.
    """

    # Dataset transforms
    transform = T.Compose([
        myT.Quantise(level_widths=cfg.LEVEL_WIDTHS),
        myT.Resize(cfg.DATA_RESCALE),
        myT.RandomResizedCrop(cfg.RESIZEDCROP_SCALE_RANGE, cfg.DATA_RESCALE),
        myT.RandomHorizontalFlip(),
        myT.ColourJitter(cfg.BRIGHTNESS, cfg.CONTRAST, cfg.SATURATION, cfg.HUE),
        myT.ToTensor()])

    # Datasets
    VOC_train = VOCDualTask(
        cfg.DSET_ROOT,
        image_set="train",
        transform=transform)
    VOC_val = VOCDualTask(
        cfg.DSET_ROOT,
        image_set="val",
        transform=transform)
    
    # Distributed samplers
    sampler_train = DistributedSampler(
        VOC_train,
        num_replicas=cfg.TRAIN_GPUS,
        rank=rank)
    sampler_val = DistributedSampler(
        VOC_val,
        num_replicas=cfg.TRAIN_GPUS,
        rank=rank)

    # Data loaders
    loader_train = DataLoader(
        VOC_train,
        batch_size=cfg.TRAIN_BATCH_SIZE,
        sampler=sampler_train,
        num_workers=cfg.DATALOADER_JOBS,
        pin_memory=True,
        drop_last=True)
    loader_val = DataLoader(
        VOC_val,
        batch_size=cfg.VAL_BATCH_SIZE,
        sampler=sampler_val,
        num_workers=cfg.DATALOADER_JOBS,
        pin_memory=True,
        drop_last=True)
    
    return loader_train, loader_val


def setup_model(rank):
    """
    Initialise model and load pretrained backbone weights.
    """
    backbone = "xception"
    
    # Initialise model
    model = Deeplabv3plus_multitask(
        seg_classes=cfg.N_CLASSES,
        dist_classes=cfg.N_ENERGY_LEVELS,
        third_branch=False,
        backbone=backbone)
    # Convert batchnorm to synchronised batchnorm
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.to(rank)
    if rank == 0:
        print(
            "GPUs found:", torch.cuda.device_count(),
            "\tGPUs used by all processes:", cfg.TRAIN_GPUS)
    # Wrap in distributed dataparallel
    model = DistributedDataParallel(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    
    # Load pretrained model weights only for backbone network, if backbone is xception
    if cfg.USE_PRETRAINED and backbone == "xception":
        current_dict = model.state_dict()
        pretrained_dict = torch.load(os.path.join(
            cfg.PRETRAINED_PATH,
            "mod_al_xception_imagenet.pth"))
        # Keys of pretrained model need to be modified so they match keys
        # of backbone in new model
        pretrained_dict = {
            "module.backbone." + k: v for k, v in pretrained_dict.items()
        }

        current_dict.update(pretrained_dict)
        model.load_state_dict(current_dict)

        if rank == 0:
            print("Loaded pretrained backbone.")
    
    if rank == 0:
        # Print number of model parameters
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Number of learnable parameters:", n_params)
    
    return model


def log_all_confusion_mat(
    logger,
    train_seg,
    train_dist,
    val_seg,
    val_dist,
    epoch,
    isnormalised):
    """
    Logs confusion matrices for segmentation and distance transform, and for 
    training and validation data (4 matrices in total).
    Args:
        - logger: tensorboard logger.
        - train_seg
        - train_dist
        - val_seg
        - val_dist
        - epoch: current training epoch.
        - isnormalised (bool): whether the matrices are normalised or not.
    """

    postfix = ""
    fmt = "0.0f"
    if isnormalised:
        postfix = "_n"
        fmt = "0.3f"

    utils.log_confusion_mat(
        logger,
        train_seg,
        (16,10),
        "train_confusion_seg" + postfix,
        fmt,
        epoch,
        list(cfg.CLASSES.values()),
        list(cfg.CLASSES.values()))
    utils.log_confusion_mat(
        logger,
        train_dist,
        (9,7),
        "train_confusion_dist" + postfix,
        fmt,
        epoch,
        "auto",
        "auto")
    utils.log_confusion_mat(
        logger,
        val_seg,
        (16,10),
        "val_confusion_seg" + postfix,
        fmt,
        epoch,
        list(cfg.CLASSES.values()),
        list(cfg.CLASSES.values()))
    utils.log_confusion_mat(
        logger,
        val_dist,
        (9,7),
        "val_confusion_dist" + postfix,
        fmt,
        epoch,
        "auto",
        "auto")


def train(rank, world_size):
    """
    Training function. Multiple processes will be spawned with this function.

    Args:
        - rank (int): index of current process.
        - world_size (int): total number of processes.
    """

    # Set address and port of master process
    # (it's required, even though I'm training on only one machine)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # Initialize process group
    distr.init_process_group("gloo", rank=rank, world_size=world_size)

    # Specify GPU to send this process to
    torch.cuda.set_device(rank)

    # Dataloaders
    loader_train, loader_val = setup_dataloaders(rank)

    # Initialise tensorboardX logger only in process 0
    if cfg.LOG and rank == 0:
        now = datetime.now()
        tbX_logger = SummaryWriter(
            os.path.join(cfg.LOG_PATH, now.strftime("%Y%m%d-%H%M")))

    # Model setup
    model = setup_model(rank)

    # Set losses for semantic, distance and gradient direction
    seg_criterion = nn.CrossEntropyLoss(ignore_index=255)

    total_px = cfg.DATA_RESCALE ** 2
    weights = torch.tensor([
        total_px / 2789,
        total_px / 6636,
        total_px / 10678,
        total_px / 26170,
        total_px / 27594,
        total_px / 33966,
        total_px / 33467,
        total_px / 40680,
        total_px / 39469,
        total_px / 40696
    ]).to(rank)
    dist_criterion = nn.CrossEntropyLoss(weight=weights)

    #grad_criterion = MeanSquaredAngularLoss()


    # Optimiser setup
    optimiser = optim.SGD(
        params = model.parameters(),
        lr = cfg.TRAIN_LR,
        momentum=cfg.TRAIN_MOMENTUM)

    # Initialise iteration index
    i = 0
    max_i = cfg.TRAIN_EPOCHS * len(loader_train)

    # Train loop
    for epoch in range(cfg.TRAIN_EPOCHS):
        
        # These values only need to be logged in process 0
        if rank == 0:
            train_loss = 0.
            train_seg_loss = 0.
            train_dist_loss = 0.
            #train_grad_loss = 0.

            if cfg.LOG:
                # Initialise confusion matrices
                train_seg_confusion = np.zeros(
                    (cfg.N_CLASSES, cfg.N_CLASSES),
                    dtype="float")
                train_dist_confusion = np.zeros(
                    (cfg.N_ENERGY_LEVELS, cfg.N_ENERGY_LEVELS),
                    dtype="float")
                val_seg_confusion = np.zeros(
                    (cfg.N_CLASSES, cfg.N_CLASSES),
                    dtype="float")
                val_dist_confusion = np.zeros(
                    (cfg.N_ENERGY_LEVELS, cfg.N_ENERGY_LEVELS),
                    dtype="float")
        
        # Initialise tqdm only on process 0
        if rank == 0:
            loader_train = tqdm(loader_train, ascii=True, desc="Train")

        model.train()

        for train_batch_i, train_batch in enumerate(loader_train):

            train_inputs = train_batch["image"].to(rank)

            # Categorical labels need to be converted from float 0-1 to integers
            train_seg = train_batch["seg"].mul(255).round().long().squeeze(1).to(rank)
            train_dist = train_batch["dist"].mul(255).round().long().squeeze(1).to(rank)

            # gradient direction is not categorical, so no conversion
            #train_grad = train_batch["grad"].to(rank)

            if cfg.ADJUST_LR:
                lr = utils.adjust_lr(
                    optimiser,
                    i,
                    max_i,
                    cfg.TRAIN_LR,
                    cfg.TRAIN_POWER)
            else:
                lr = cfg.TRAIN_LR

            optimiser.zero_grad()

            train_predicted_seg, train_predicted_dist = model(train_inputs)

            # Calculate losses
            seg_loss = seg_criterion(train_predicted_seg, train_seg)
            dist_loss = dist_criterion(train_predicted_dist, train_dist)
            #grad_loss = grad_criterion(train_predicted_grad, train_grad)
            loss = seg_loss + dist_loss # + grad_loss
            loss.backward()
            optimiser.step()

            i += 1

            # Update losses to display
            if rank == 0:
                train_loss += loss.item()
                train_seg_loss += seg_loss.item()
                train_dist_loss += dist_loss.item()
                #train_grad_loss += grad_loss.item()

                batch_pixels = cfg.DATA_RESCALE ** 2 * cfg.TRAIN_BATCH_SIZE
                seg_argmax = torch.argmax(train_predicted_seg, dim=1)
                dist_argmax = torch.argmax(train_predicted_dist, dim=1)

                if cfg.LOG and epoch % 20 == 0 or epoch == cfg.TRAIN_EPOCHS - 1:
                    # Accumulate confusion matrices
                    train_seg_confusion += calc_confusion_matrix(
                        train_seg.cpu(),
                        seg_argmax.cpu(),
                        labels=list(cfg.CLASSES.keys())
                        ) / cfg.TRAIN_BATCH_SIZE
                    train_dist_confusion += calc_confusion_matrix(
                        train_dist.cpu(),
                        dist_argmax.cpu(),
                        labels=list(range(cfg.N_ENERGY_LEVELS))
                        ) / cfg.TRAIN_BATCH_SIZE

        # Again, only log on process 0
        if rank == 0:
            val_loss = 0.
            val_seg_loss = 0.
            val_dist_loss = 0.
            #val_grad_loss = 0.

        # Initialise tqdm
        if rank == 0:
            loader_val = tqdm(loader_val, ascii=True, desc="Valid")

        model.eval()

        with torch.no_grad():
            for val_batch_i, val_batch in enumerate(loader_val):

                val_inputs = val_batch["image"].to(rank)
                val_seg = val_batch["seg"].mul(255).round().long().squeeze(1).to(rank)
                val_dist = val_batch["dist"].mul(255).round().long().squeeze(1).to(rank)
                #val_grad = val_batch["grad"].to(rank)

                val_predicted_seg, val_predicted_dist = model(val_inputs)

                # Calculate losses
                if rank == 0:
                    seg_loss = seg_criterion(val_predicted_seg, val_seg)
                    dist_loss = dist_criterion(val_predicted_dist, val_dist)
                    #grad_loss = grad_criterion(val_predicted_grad, val_grad)
                    loss = seg_loss + dist_loss # + grad_loss
                    val_loss += loss.item()
                    val_seg_loss += seg_loss.item()
                    val_dist_loss += dist_loss.item()
                    #val_grad_loss += grad_loss.item()

                    batch_pixels = cfg.DATA_RESCALE ** 2 * cfg.VAL_BATCH_SIZE
                    seg_argmax = torch.argmax(val_predicted_seg, dim=1)
                    dist_argmax = torch.argmax(val_predicted_dist, dim=1)

                    if cfg.LOG and epoch % 20 == 0 or epoch == cfg.TRAIN_EPOCHS - 1:
                        # Accumulate confusion matrices
                        val_seg_confusion += calc_confusion_matrix(
                            val_seg.cpu(),
                            seg_argmax.cpu(),
                            labels=list(cfg.CLASSES.keys())
                            ) / cfg.TRAIN_BATCH_SIZE
                        val_dist_confusion += calc_confusion_matrix(
                            val_dist.cpu(),
                            dist_argmax.cpu(),
                            labels=list(range(cfg.N_ENERGY_LEVELS))
                            ) / cfg.TRAIN_BATCH_SIZE

        # Only print info and log to tensorboard in process 0
        if rank == 0:
            # Average training losses on all batches
            train_loss /= len(loader_train)
            train_seg_loss /= len(loader_train)
            train_dist_loss /= len(loader_train)
            #train_grad_loss /= len(loader_train)

            # Average validation losses on batches
            val_loss /= len(loader_val)
            val_seg_loss /= len(loader_val)
            val_dist_loss /= len(loader_val)
            #val_grad_loss /= len(loader_val)

            print("Epoch: %d/%d\ti: %d\tlr: %g\ttrain_loss: %g\tval_loss: %g\n"
                % (epoch+1, cfg.TRAIN_EPOCHS, i, lr, train_loss, val_loss))

            if cfg.LOG:
                # Log scalars to tensorboardX
                tbX_logger.add_scalars(
                    "total_losses", {
                        "train_loss": train_loss,
                        "val_loss": val_loss
                    },
                    epoch)
                tbX_logger.add_scalars(
                    "seg_loss", {  
                        "train_seg_loss": train_seg_loss,
                        "val_seg_loss": val_seg_loss
                    },
                    epoch)
                tbX_logger.add_scalars(
                    "dist_loss", {
                        "train_dist_loss": train_dist_loss,
                        "val_dist_loss": val_dist_loss
                    },
                    epoch)
                #tbX_logger.add_scalars(
                #    "grad_loss", {
                #        "train_grad_loss": train_grad_loss,
                #        "val_grad_loss": val_grad_loss
                #    },
                #    epoch)
                                                 
                tbX_logger.add_scalar("lr", lr, epoch)

                # it seems like tensorboard doesn't like saving a lot of images,
                # so log images only every few epochs
                if epoch % 20 == 0 or epoch == cfg.TRAIN_EPOCHS - 1:


                    # Calculate IoUs from confusion matrices
                    train_seg_iou = calc_iou(train_seg_confusion)
                    train_dist_iou = calc_iou(train_dist_confusion)
                    val_seg_iou = calc_iou(val_seg_confusion)
                    val_dist_iou = calc_iou(val_dist_confusion)
                    
                    tbX_logger.add_scalars(
                        "seg_iou", {
                            "train_seg_iou": np.mean(train_seg_iou),
                            "val_seg_iou": np.mean(val_seg_iou)
                        },
                        epoch)
                    tbX_logger.add_scalars(
                        "dist_iou", {
                            "train_dist_iou": np.mean(train_dist_iou),
                            "val_dist_iou": np.mean(val_dist_iou)
                        },
                        epoch)

                    # Convert training images for plotting
                    train_input_tb = make_grid(train_inputs).cpu().numpy()
                    train_seg_tb = make_grid(utils.colormap(
                        train_seg.float().div(cfg.N_CLASSES).unsqueeze(1).cpu())).numpy()
                    train_seg_prediction_tb = make_grid(utils.colormap(
                        torch.argmax(train_predicted_seg, dim=1).float() \
                        .div(cfg.N_CLASSES).unsqueeze(1).cpu())).numpy()
                    train_dist_tb = make_grid(utils.colormap(
                        train_dist.float().div(cfg.N_ENERGY_LEVELS).unsqueeze(1).cpu())).numpy()
                    train_dist_prediction_tb = make_grid(utils.colormap(
                        torch.argmax(train_predicted_dist, dim=1).float() \
                        .div(cfg.N_ENERGY_LEVELS).unsqueeze(1).cpu())).numpy()

                    blue_channel = torch.zeros(
                        cfg.TRAIN_BATCH_SIZE,
                        1,
                        cfg.DATA_RESCALE,
                        cfg.DATA_RESCALE).to(rank)
                    # Normalise grad vectors for image drawing
                    #grad_norm = torch.norm(train_predicted_grad, p=2, dim=1, keepdim=True)
                    #train_predicted_grad = train_predicted_grad.div(grad_norm)
                    #train_grad_tb = make_grid(torch.cat(
                    #    (train_grad, blue_channel), 
                    #    dim=1)).cpu().numpy()
                    #train_grad_prediction_tb = make_grid(torch.cat(
                    #    (train_predicted_grad, blue_channel),
                    #    dim=1)).cpu().detach().numpy()                          

                    # Convert val images for plotting
                    val_input_tb = make_grid(val_inputs).cpu().numpy()
                    val_seg_tb = make_grid(utils.colormap(
                        val_seg.float().div(cfg.N_CLASSES).unsqueeze(1).cpu())).numpy()
                    val_seg_prediction_tb = make_grid(utils.colormap(
                        torch.argmax(val_predicted_seg, dim=1).float() \
                        .div(cfg.N_CLASSES).unsqueeze(1).cpu())).numpy()
                    val_dist_tb = make_grid(utils.colormap(
                        val_dist.float().div(cfg.N_ENERGY_LEVELS).unsqueeze(1).cpu())).numpy()
                    val_dist_prediction_tb = make_grid(utils.colormap(
                        torch.argmax(val_predicted_dist, dim=1).float() \
                        .div(cfg.N_ENERGY_LEVELS).unsqueeze(1).cpu())).numpy()
                    # Normalise grad vectors for image drawing
                    #grad_norm = torch.norm(val_predicted_grad, p=2, dim=1, keepdim=True)
                    #val_predicted_grad = val_predicted_grad.div(grad_norm)
                    #val_grad_tb = make_grid(torch.cat(
                    #    (val_grad, blue_channel), 
                    #    dim=1)).cpu().numpy()
                    #val_grad_prediction_tb = make_grid(torch.cat(
                    #    (val_predicted_grad, blue_channel),
                    #    dim=1)).cpu().detach().numpy()                          

                    # Training images
                    tbX_logger.add_image("train_input", train_input_tb, epoch)
                    tbX_logger.add_image("train_seg", train_seg_tb, epoch)
                    tbX_logger.add_image(
                        "train_seg_prediction",
                        train_seg_prediction_tb,
                        epoch)
                    tbX_logger.add_image("train_dist", train_dist_tb, epoch)    
                    tbX_logger.add_image(
                        "train_dist_prediction",
                        train_dist_prediction_tb,
                        epoch)
                    #tbX_logger.add_image("train_grad", train_grad_tb, epoch)
                    #tbX_logger.add_image(
                    #    "train_grad_prediction",
                    #    train_grad_prediction_tb,
                    #    epoch)
                    
                    # Validation images
                    tbX_logger.add_image("val_input", val_input_tb, epoch)
                    tbX_logger.add_image("val_seg", val_seg_tb, epoch)
                    tbX_logger.add_image(
                        "val_seg_prediction",
                        val_seg_prediction_tb,
                        epoch)
                    tbX_logger.add_image("val_dist", val_dist_tb, epoch)    
                    tbX_logger.add_image(
                        "val_dist_prediction",
                        val_dist_prediction_tb,
                        epoch)
                    #tbX_logger.add_image("val_grad", val_grad_tb, epoch)
                    #tbX_logger.add_image(
                    #    "val_grad_prediction",
                    #    val_grad_prediction_tb,
                    #    epoch)

                    # Divide unnormalised matrices
                    train_seg_confusion /= len(loader_train)
                    train_dist_confusion /= len(loader_train)
                    val_seg_confusion /= len(loader_val)
                    val_dist_confusion /= len(loader_val)

                    # Normalise confusion matrices
                    train_seg_confusion_n = utils.normalise_confusion_mat(
                        train_seg_confusion)
                    train_dist_confusion_n = utils.normalise_confusion_mat(
                        train_dist_confusion)
                    val_seg_confusion_n = utils.normalise_confusion_mat(
                        val_seg_confusion)
                    val_dist_confusion_n = utils.normalise_confusion_mat(
                        val_dist_confusion)

                    # Log confusion matrices to tensorboard
                    log_all_confusion_mat(
                        tbX_logger,
                        train_seg_confusion,
                        train_dist_confusion,
                        val_seg_confusion,
                        val_dist_confusion,
                        epoch,
                        isnormalised=False)

                    # Log normalised confusion matrices to tensorboard
                    log_all_confusion_mat(
                        tbX_logger,
                        train_seg_confusion_n,
                        train_dist_confusion_n,
                        val_seg_confusion_n,
                        val_dist_confusion_n,
                        epoch,
                        isnormalised=True)

        # Save checkpoint
        if epoch % 20 == 0 and epoch != 0 and rank == 0:
            save_path = os.path.join(
                cfg.PRETRAINED_PATH,
                "%s_%s_%s_epoch%d.pth"
                %(cfg.MODEL_NAME, cfg.MODEL_BACKBONE, cfg.DATA_NAME, epoch))
            torch.save(model.state_dict(), save_path)
            print("%s has been saved." %save_path)

    if cfg.LOG and rank == 0:
        tbX_logger.close()

    # Save final trained model
    if rank == 0:
        save_path = os.path.join(
            cfg.PRETRAINED_PATH,
            "%s_%s_%s_epoch%d_final_noskip_noaspp.pth"
            %(cfg.MODEL_NAME, cfg.MODEL_BACKBONE, cfg.DATA_NAME, cfg.TRAIN_EPOCHS))
        torch.save(model.state_dict(), save_path)
        print("FINISHED: %s has been saved." %save_path)

    distr.destroy_process_group()


if __name__ == "__main__":

    world_size = cfg.TRAIN_GPUS
    # Spawn training processes
    multiproc.spawn(train, args=(world_size,), nprocs=world_size, join=True)