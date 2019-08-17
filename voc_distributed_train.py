from datetime import datetime
from itertools import chain
import os
from pprint import pprint
import random
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

from models.deeplabv3plus_multitask import Deeplabv3plus_multitask
import transforms.transforms as myT
from config.config import VOCConfig
import utils.helpers as helpers


cfg = VOCConfig()


#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


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
    Initialise model and load weights.
    """
    
    # Initialise model
    model = Deeplabv3plus_multitask(
        seg_classes=cfg.N_CLASSES,
        dist_classes=cfg.N_ENERGY_LEVELS)
    # Convert batchnorm to synchronised batchnorm
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.to(rank)
    if rank == 0:
        print(
            "GPUs found:", torch.cuda.device_count(),
            "\tGPUs used by all processes:", cfg.TRAIN_GPUS)
    # Wrap in distributed dataparallel
    model = DistributedDataParallel(model, device_ids=[rank], output_device=rank)

    # Load pretrained model weights only for backbone network
    if cfg.USE_PRETRAINED:
        current_dict = model.state_dict()
        pretrained_dict = torch.load(
            os.path.join(
                cfg.PRETRAINED_PATH,
                "deeplabv3plus_xception_VOC2012_epoch46_all.pth"))
        pretrained_dict = {
            k: v for k, v in pretrained_dict.items()
            if "backbone" in k and k in current_dict
        }
        current_dict.update(pretrained_dict)
        model.load_state_dict(current_dict)
    
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

    helpers.log_confusion_mat(
        logger,
        train_seg,
        (16,10),
        "train_confusion_seg" + postfix,
        fmt,
        epoch,
        list(cfg.CLASSES.values()),
        list(cfg.CLASSES.values()))
    helpers.log_confusion_mat(
        logger,
        train_dist,
        (9,7),
        "train_confusion_dist" + postfix,
        fmt,
        epoch,
        "auto",
        "auto")
    helpers.log_confusion_mat(
        logger,
        val_seg,
        (16,10),
        "val_confusion_seg" + postfix,
        fmt,
        epoch,
        list(cfg.CLASSES.values()),
        list(cfg.CLASSES.values()))
    helpers.log_confusion_mat(
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

    # Fix all seeds for reproducibility
    seed = 777
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False

    # Dataloaders
    loader_train, loader_val = setup_dataloaders(rank)

    # Initialise tensorboardX logger only in process 0
    if cfg.LOG and rank == 0:
        now = datetime.now()
        tbX_logger = SummaryWriter(
            os.path.join(cfg.LOG_PATH, now.strftime("%Y%m%d-%H%M")))

    # Model setup
    model = setup_model(rank)

    # Set losses (first one for semantics, second one for watershed)
    seg_criterion = nn.CrossEntropyLoss(ignore_index=255)
    weights = torch.tensor(261500).div(torch.tensor([
        14000, 25500, 23000, 25000, 24000,
        22500, 22500, 22000, 24000, 58500],
        dtype=torch.float)) \
        .to(rank)
    dist_criterion = nn.CrossEntropyLoss(weight=weights)

    # Optimiser setup
    optimiser = optim.SGD(
        params = [
            {'params': helpers.get_params(model, key='1x'), 'lr': cfg.TRAIN_LR},
            {'params': helpers.get_params(model, key='10x'), 'lr': 10 * cfg.TRAIN_LR}
        ],
        momentum=cfg.TRAIN_MOMENTUM)

    # Initialise iteration index
    i = 0
    max_i = cfg.TRAIN_EPOCHS * len(loader_train)

    # Train loop
    for epoch in range(cfg.TRAIN_EPOCHS):
        
        # These values only need to be logged in process 0
        if rank == 0:
            train_loss = 0.0
            train_seg_loss = 0.0
            train_dist_loss = 0.0
            train_seg_acc = 0.0
            train_dist_acc = 0.0

        # Only initialise confusion matrices when they're going to be logged
        # Also, only log from process 0
        if epoch == cfg.TRAIN_EPOCHS - 1 and cfg.LOG and rank == 0:
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
            # Labels need to be converted from float 0-1 to integers
            train_seg = train_batch["seg"].mul(255).round().long().squeeze(1).to(rank)
            train_dist = train_batch["dist"].mul(255).round().long().squeeze(1).to(rank)

            if cfg.ADJUST_LR:
                lr = helpers.adjust_lr(
                    optimiser,
                    i, max_i,
                    cfg.TRAIN_LR,
                    cfg.TRAIN_POWER)
            else:
                lr = cfg.TRAIN_LR

            optimiser.zero_grad()

            train_predicted_seg, train_predicted_dist = model(train_inputs)

            # Calculate losses
            seg_loss = seg_criterion(train_predicted_seg, train_seg)
            dist_loss = dist_criterion(train_predicted_dist, train_dist)
            loss = seg_loss + dist_loss
            loss.backward()
            optimiser.step()

            i += 1

            # Update losses to display
            if rank == 0:
                train_loss += loss.item()
                train_seg_loss += seg_loss.item()
                train_dist_loss += dist_loss.item()

                # Accumulate accuracies
                batch_pixels = cfg.DATA_RESCALE ** 2 * cfg.TRAIN_BATCH_SIZE
                seg_argmax = torch.argmax(train_predicted_seg, dim=1)
                dist_argmax = torch.argmax(train_predicted_dist, dim=1)
                train_seg_acc += torch.sum(train_seg == seg_argmax) \
                    .float().div(batch_pixels)
                train_dist_acc += torch.sum(train_dist == dist_argmax) \
                    .float().div(batch_pixels)

                # Only calculate confusion matrices every few epochs
                if epoch == cfg.TRAIN_EPOCHS - 1 and cfg.LOG:

                    # Flatten labels and predictions and append
                    total_seg_labels = train_seg.cpu().flatten()
                    total_seg_predictions = seg_argmax.cpu().flatten()

                    total_dist_labels = train_dist.cpu().flatten()
                    total_dist_predictions = dist_argmax.cpu().flatten()

                    # Accumulate confusion matrix 
                    train_seg_confusion += confusion_matrix(
                        total_seg_labels,
                        total_seg_predictions,
                        labels=list(cfg.CLASSES.keys())
                        ) / cfg.TRAIN_BATCH_SIZE

                    train_dist_confusion += confusion_matrix(
                        total_dist_labels,
                        total_dist_predictions,
                        labels=list(range(cfg.N_ENERGY_LEVELS))
                        ) / cfg.TRAIN_BATCH_SIZE

        # Again, only log on process 0
        if rank == 0:
            val_loss = 0.0
            val_seg_loss = 0.0
            val_dist_loss = 0.0
            val_seg_acc = 0.0
            val_dist_acc = 0.0

        # Initialise tqdm
        if rank == 0:
            loader_val = tqdm(loader_val, ascii=True, desc="Valid")

        model.eval()

        with torch.no_grad():
            for val_batch_i, val_batch in enumerate(loader_val):

                val_inputs = val_batch["image"].to(rank)
                val_seg = val_batch["seg"].mul(255).round().long().squeeze(1).to(rank)
                val_dist = val_batch["dist"].mul(255).round().long().squeeze(1).to(rank)

                val_predicted_seg, val_predicted_dist = model(val_inputs)

                # Calculate losses
                if rank == 0:
                    seg_loss = seg_criterion(val_predicted_seg, val_seg)
                    dist_loss = dist_criterion(val_predicted_dist, val_dist)
                    loss = seg_loss + dist_loss
                    val_loss += loss.item()
                    val_seg_loss += seg_loss.item()
                    val_dist_loss += dist_loss.item()

                    # Calculate accuracies
                    batch_pixels = cfg.DATA_RESCALE ** 2 * cfg.VAL_BATCH_SIZE
                    seg_argmax = torch.argmax(val_predicted_seg, dim=1)
                    dist_argmax = torch.argmax(val_predicted_dist, dim=1)
                    val_seg_acc += torch.sum(val_seg == seg_argmax) \
                        .float().div(batch_pixels)
                    val_dist_acc += torch.sum(val_dist == dist_argmax) \
                        .float().div(batch_pixels)

                    if epoch == cfg.TRAIN_EPOCHS - 1 and cfg.LOG:
                        # Flatten labels and predictions and append
                        total_seg_labels = val_seg.cpu().flatten()
                        total_seg_predictions = seg_argmax.cpu().flatten()

                        total_dist_labels = val_dist.cpu().flatten()
                        total_dist_predictions = dist_argmax.cpu().flatten()

                        # Accumulate confusion matrix 
                        val_seg_confusion += confusion_matrix(
                            total_seg_labels,
                            total_seg_predictions,
                            labels=list(cfg.CLASSES.keys())
                            ) / cfg.VAL_BATCH_SIZE

                        val_dist_confusion += confusion_matrix(
                            total_dist_labels,
                            total_dist_predictions,
                            labels=list(range(cfg.N_ENERGY_LEVELS))
                            ) / cfg.VAL_BATCH_SIZE

        # Only print info and log to tensorboard in process 0
        if rank == 0:
            # Average training losses and acc on all batches
            train_loss /= len(loader_train)
            train_seg_loss /= len(loader_train)
            train_dist_loss /= len(loader_train)
            train_seg_acc /= len(loader_train)
            train_dist_acc /= len(loader_train)

            # Average validation losses and acc on batches
            val_loss /= len(loader_val)
            val_seg_loss /= len(loader_val)
            val_dist_loss /= len(loader_val)
            val_seg_acc /= len(loader_val)
            val_dist_acc /= len(loader_val)

            print("Epoch: %d/%d\ti: %d\tlr: %g\ttrain_loss: %g\tval_loss: %g\n"
                % (epoch+1, cfg.TRAIN_EPOCHS, i, lr, train_loss, val_loss))

            if cfg.LOG:
                # Convert training data for plotting
                train_input_tb = make_grid(train_inputs).cpu().numpy()
                train_seg_tb = make_grid(helpers.colormap(
                    train_seg.float().div(cfg.N_CLASSES).unsqueeze(1).cpu())).numpy()
                train_seg_prediction_tb = make_grid(helpers.colormap(
                    torch.argmax(train_predicted_seg, dim=1).float() \
                    .div(cfg.N_CLASSES).unsqueeze(1).cpu())).numpy()
                train_dist_tb = make_grid(helpers.colormap(
                    train_dist.float().div(cfg.N_ENERGY_LEVELS).unsqueeze(1).cpu())).numpy()
                train_dist_prediction_tb = make_grid(helpers.colormap(
                    torch.argmax(train_predicted_dist, dim=1).float() \
                    .div(cfg.N_ENERGY_LEVELS).unsqueeze(1).cpu())).numpy()                                         

                # Convert val data for plotting
                val_input_tb = make_grid(val_inputs).cpu().numpy()
                val_seg_tb = make_grid(helpers.colormap(
                    val_seg.float().div(cfg.N_CLASSES).unsqueeze(1).cpu())).numpy()
                val_seg_prediction_tb = make_grid(helpers.colormap(
                    torch.argmax(val_predicted_seg, dim=1).float() \
                    .div(cfg.N_CLASSES).unsqueeze(1).cpu())).numpy()
                val_dist_tb = make_grid(helpers.colormap(
                    val_dist.float().div(cfg.N_ENERGY_LEVELS).unsqueeze(1).cpu())).numpy()
                val_dist_prediction_tb = make_grid(helpers.colormap(
                    torch.argmax(val_predicted_dist, dim=1).float() \
                    .div(cfg.N_ENERGY_LEVELS).unsqueeze(1).cpu())).numpy()

                # Log scalars to tensorboardX
                tbX_logger.add_scalars(
                    "total_losses",
                    {
                        "train_loss": train_loss,
                        "val_loss": val_loss
                    },
                    epoch)
                tbX_logger.add_scalars(
                    "seg_losses",
                    {  
                        "train_seg_loss": train_seg_loss,
                        "val_seg_loss": val_seg_loss
                    },
                    epoch)
                tbX_logger.add_scalars(
                    "dist_losses",
                    {
                        "train_dist_loss": train_dist_loss,
                        "val_dist_loss": val_dist_loss
                    },
                    epoch)
                tbX_logger.add_scalars(
                    "seg_acc",
                    {
                        "train_seg_acc": train_seg_acc,
                        "val_seg_acc": val_seg_acc
                    },
                    epoch)
                tbX_logger.add_scalars(
                    "dist_acc",
                    {
                        "train_dist_acc": train_dist_acc,
                        "val_dist_acc": val_dist_acc
                    },
                    epoch)                                 
                tbX_logger.add_scalar("lr", lr, epoch)

                # it seems like tensorboard doesn't like saving a lot of images,
                # so log images only at last epoch
                if epoch == cfg.TRAIN_EPOCHS - 1:

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

                    # Divide unnormalised matrices
                    train_seg_confusion /= len(loader_train)
                    train_dist_confusion /= len(loader_train)
                    val_seg_confusion /= len(loader_val)
                    val_dist_confusion /= len(loader_val)

                    # Normalise confusion matrices
                    train_seg_confusion_n = helpers.normalise_confusion_mat(
                        train_seg_confusion)
                    train_dist_confusion_n = helpers.normalise_confusion_mat(
                        train_dist_confusion)
                    val_seg_confusion_n = helpers.normalise_confusion_mat(
                        val_seg_confusion)
                    val_dist_confusion_n = helpers.normalise_confusion_mat(
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
        if epoch % 100 == 0 and epoch != 0 and rank == 0:
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
            "%s_%s_%s_epoch%d_final.pth"
            %(cfg.MODEL_NAME, cfg.MODEL_BACKBONE, cfg.DATA_NAME, cfg.TRAIN_EPOCHS))
        torch.save(model.state_dict(), save_path)
        print("FINISHED: %s has been saved." %save_path)

    distr.destroy_process_group()


if __name__ == "__main__":

    world_size = cfg.TRAIN_GPUS
    # Spawn training processes
    multiproc.spawn(train, args=(world_size,), nprocs=world_size, join=True)