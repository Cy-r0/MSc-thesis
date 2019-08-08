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
from tensorboardX.utils import figure_to_image
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


#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"



# Constants here
LOG = True
LOG_PATH = "logs"

CLASSES = ["aeroplane",
           "bicycle",
           "bird",
           "boat",
           "bottle",
           "bus",
           "car",
           "cat",
           "chair",
           "cow",
           "diningtable",
           "dog",
           "horse",
           "motorbike",
           "person",
           "pottedplant",
           "sheep",
           "sofa",
           "train",
           "tvmonitor"]
SEG_CLASSES = len(CLASSES) + 1
DATALOADER_JOBS = 4
DSET_ROOT = "/home/cyrus/Datasets"

DATA_RESCALE = 512
DATA_RANDOMCROP = 512

TRAIN_GPUS = 2
TEST_GPUS = 2

RESUME = True
PRETRAINED_PATH = "models/pretrained"

TRAIN_BATCH_SIZE = 8
TEST_BATCH_SIZE = 8
TRAIN_LR = 0.007
TRAIN_POWER = 0.9
TRAIN_MOMENTUM = 0.9
TRAIN_EPOCHS = 45
VAL_FRACTION = 0.5
ADJUST_LR = True
LEVEL_WIDTHS = [1,5,6,8,9,10,12,14,20]
ENERGY_LEVELS = len(LEVEL_WIDTHS) + 1

MODEL_NAME = "deeplabv3plus_multitask"
MODEL_BACKBONE = 'xception'
DATA_NAME = 'VOC2012'

IMG_LOG_EPOCHS = 10


# Helper methods

def adjust_lr(optimizer, i, max_i):
    """
    Gradually decrease learning rate as iterations increase.
    """
    lr = TRAIN_LR * (1 - i/(max_i + 1)) ** TRAIN_POWER
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

def log_confusion_mat(confusion_mat, figsize, title, fmt, epoch):
    """
    Log confusion matrix to tensorboard as matplotlib figure.
    """
    fig = plt.figure(figsize=figsize)
    sn.heatmap(pd.DataFrame(confusion_mat), annot=True, fmt=fmt)
    plt.ylabel("True")
    plt.xlabel("Predicted")
    confusion_img = figure_to_image(fig, close=True)
    tbX_logger.add_image(title, confusion_img, epoch)

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
        - segs (4D ndarray of uint8).
        - dists (3D ndarray of uint8).
        - energy_cut (int): energy level to binarize image at.
        - min_area (int): minimum area of a contour in pixels.
    """

    for seg, dist in zip(segs, dists):

        # This block is super fast (0.5 ms)
        _, thres = cv2.threshold(np.copy(dist), energy_cut, 255, cv2.THRESH_BINARY)
        _, contours, _ = cv2.findContours(np.copy(thres), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        contoured = np.copy(dist)

        # This list comprehension is quite expensive (8 ms)
        contours = [c for c in contours if cv2.contourArea(c) >= min_area]

        for contour in contours:

            for seg_class in seg:

                # Create binary mask from contour

                # Mask && seg class to select only pixels inside mask

                # 


        cv2.drawContours(contoured, contours, -1, 255, 1)
        cv2.imshow("img", contoured)
        cv2.waitKey(0)


def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation="nearest")


# Fix all seeds for reproducibility
np.random.seed(777)
torch.manual_seed(777)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Dataset transforms
transform = T.Compose([myT.Quantise(level_widths=LEVEL_WIDTHS),
                       myT.Resize(DATA_RESCALE),
                       myT.RandomCrop(DATA_RANDOMCROP),
                       myT.ToTensor(),
                       #myT.Normalise((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ])

# Dataset setup
VOCW_train = VOCDualTask(DSET_ROOT,
                         image_set="train_reduced",
                         transform=transform)
VOCW_val = VOCDualTask(DSET_ROOT,
                       image_set="val_reduced",
                       transform=transform)

loader_train = DataLoader(VOCW_train,
                          batch_size = TRAIN_BATCH_SIZE,
                          sampler=sampler.SubsetRandomSampler(
                              range(int(len(VOCW_train) * (1 - VAL_FRACTION)))),
                          num_workers = DATALOADER_JOBS,
                          pin_memory = True,
                          drop_last=True)
loader_val = DataLoader(VOCW_train,
                        batch_size = TEST_BATCH_SIZE,
                        sampler=sampler.SubsetRandomSampler(
                            range(int(len(VOCW_train) * (1 - VAL_FRACTION)),
                                  len(VOCW_train))),
                        num_workers = DATALOADER_JOBS,
                        pin_memory = True,
                        drop_last=True)
loader_test = DataLoader(VOCW_val,
)


# Initialise tensorboardX logger
now = datetime.now()
if LOG:
    tbX_logger = SummaryWriter(os.path.join(LOG_PATH,
                                            now.strftime("%Y%m%d-%H%M")))

# Model setup
model = Deeplabv3plus_multitask(seg_classes=SEG_CLASSES,
                                dist_classes=ENERGY_LEVELS)

# Move model to GPU devices
device = torch.device(0)
print("GPUs found:", torch.cuda.device_count(), "\tGPUs used:", TRAIN_GPUS)

if TRAIN_GPUS > 1:
    model = nn.DataParallel(model)

# Load pretrained model weights only for backbone network
if RESUME:
    current_dict = model.state_dict()
    pretrained_dict = torch.load(os.path.join(PRETRAINED_PATH,
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
            {'params': get_params(model, key='1x'), 'lr': TRAIN_LR},
            {'params': get_params(model, key='10x'), 'lr': 10*TRAIN_LR}
        ],
        momentum=TRAIN_MOMENTUM)

counts = [0] * ENERGY_LEVELS

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)



# Training loop
i = 0
max_i = TRAIN_EPOCHS * len(loader_train)
for epoch in range(TRAIN_EPOCHS):

    train_loss = 0.0
    train_seg_loss = 0.0
    train_dist_loss = 0.0
    train_seg_acc = 0.0
    train_dist_acc = 0.0

    # Only initialise confusion matrices when they're going to be logged
    if epoch % IMG_LOG_EPOCHS == 0:
        train_seg_confusion = np.zeros((SEG_CLASSES + 1, SEG_CLASSES + 1), dtype="float")
        train_dist_confusion = np.zeros((ENERGY_LEVELS, ENERGY_LEVELS), dtype="float")
        val_seg_confusion = np.zeros((SEG_CLASSES + 1, SEG_CLASSES + 1), dtype="float")
        val_dist_confusion = np.zeros((ENERGY_LEVELS, ENERGY_LEVELS), dtype="float")
    
    # Initialise tqdm
    tqdm_loader_train = tqdm(loader_train, ascii=True, desc="Train")

    model.train()
    for train_batch_i, train_batch in enumerate(tqdm_loader_train):

        train_inputs = train_batch["image"].to(device)
        # Labels need to be converted from float 0-1 to integers
        train_seg = train_batch["seg"].mul(255).round().long().squeeze(1).to(device)
        train_dist = train_batch["dist"].mul(255).round().long().squeeze(1).to(device)

        if ADJUST_LR:
            lr = adjust_lr(optimiser, i, max_i)
        else:
            lr = TRAIN_LR

        optimiser.zero_grad()

        train_predicted_seg, train_predicted_dist = model(train_inputs)

        if epoch == TRAIN_EPOCHS-1:
            # Convert images to uint8 before postprocessing
            pp_seg = torch.argmax(train_predicted_seg, dim=1).cpu().byte().numpy()
            pp_dist = torch.argmax(train_predicted_dist, dim=1).cpu().byte().numpy()
            postprocess(pp_seg, pp_dist, energy_cut=1, min_area=int(DATA_RESCALE ** 2 * 0.0005))

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
                          .float().div(DATA_RESCALE ** 2 * TRAIN_BATCH_SIZE)).data
        train_dist_acc += (torch.sum(train_dist == torch.argmax(train_predicted_dist, dim=1))
                           .float().div(DATA_RESCALE ** 2 * TRAIN_BATCH_SIZE)).data

        # Only calculate confusion matrices every few epochs
        if epoch % IMG_LOG_EPOCHS == 0:

            total_seg_labels = train_seg.cpu().flatten()
            total_seg_predictions = torch.argmax(train_predicted_seg, dim=1).cpu().flatten()
            train_seg_confusion += confusion_matrix(total_seg_labels, total_seg_predictions,
                                                    labels=list(chain(range(SEG_CLASSES), [255])))
            train_seg_confusion /= TRAIN_BATCH_SIZE

            total_dist_labels = train_dist.cpu().flatten()
            total_dist_predictions = torch.argmax(train_predicted_dist, dim=1).cpu().flatten()
            train_dist_confusion += confusion_matrix(total_dist_labels, total_dist_predictions,
                                                        labels=list(range(ENERGY_LEVELS)))
            train_dist_confusion /= TRAIN_BATCH_SIZE

        # Accumulate pixel belonging to each class (for weighted loss)
        #for class_i in range(ENERGY_LEVELS):
         #   counts[class_i] += torch.nonzero(train_labels.flatten() == class_i).flatten().size(0)

    # Print counts
    #counts = [c / (len(loader_train) * TRAIN_BATCH_SIZE) for c in counts]
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
                            .float().div(DATA_RESCALE ** 2 * TRAIN_BATCH_SIZE)).data
            val_dist_acc += (torch.sum(val_dist == torch.argmax(val_predicted_dist, dim=1))
                             .float().div(DATA_RESCALE ** 2 * TRAIN_BATCH_SIZE)).data

            if epoch % IMG_LOG_EPOCHS == 0:
                # Accumulate labels and predictions for confusion matrices
                total_seg_labels = val_seg.cpu().flatten()
                total_seg_predictions = torch.argmax(val_predicted_seg, dim=1).cpu().flatten()
                val_seg_confusion += confusion_matrix(total_seg_labels, total_seg_predictions,
                                                        labels=list(chain(range(SEG_CLASSES), [255])))
                val_seg_confusion /= TEST_BATCH_SIZE

                total_dist_labels = val_dist.cpu().flatten()
                total_dist_predictions = torch.argmax(val_predicted_dist, dim=1).cpu().flatten()
                val_dist_confusion += confusion_matrix(total_dist_labels, total_dist_predictions,
                                                        labels=list(range(ENERGY_LEVELS)))
                val_dist_confusion /= TEST_BATCH_SIZE

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
          % (epoch+1, TRAIN_EPOCHS, i, lr, train_loss, val_loss))

    # Convert training data for plotting
    train_input_tb = make_grid(train_inputs).cpu().numpy()
    train_seg_tb = make_grid(colormap(train_seg.float().div(SEG_CLASSES)
                                        .unsqueeze(1).cpu())).numpy()
    train_seg_prediction_tb = make_grid(colormap(torch.argmax(train_predicted_seg, dim=1)
                                              .float().div(SEG_CLASSES).unsqueeze(1)
                                              .cpu())).numpy()
    train_dist_tb = make_grid(colormap(train_dist.float().div(ENERGY_LEVELS)
                                        .unsqueeze(1).cpu())).numpy()
    train_dist_prediction_tb = make_grid(colormap(torch.argmax(train_predicted_dist, dim=1)
                                              .float().div(ENERGY_LEVELS).unsqueeze(1)
                                              .cpu())).numpy()                                         

    # Convert val data for plotting
    val_input_tb = make_grid(val_inputs).cpu().numpy()
    val_seg_tb = make_grid(colormap(val_seg.float().div(SEG_CLASSES)
                                      .unsqueeze(1).cpu())).numpy()
    val_seg_prediction_tb = make_grid(colormap(torch.argmax(val_predicted_seg, dim=1)
                                            .float().div(SEG_CLASSES).unsqueeze(1)
                                            .cpu())).numpy()
    val_dist_tb = make_grid(colormap(val_dist.float().div(ENERGY_LEVELS)
                                        .unsqueeze(1).cpu())).numpy()
    val_dist_prediction_tb = make_grid(colormap(torch.argmax(val_predicted_dist, dim=1)
                                              .float().div(ENERGY_LEVELS).unsqueeze(1)
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
    if epoch % IMG_LOG_EPOCHS == 0:

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
        train_seg_confusion_n = normalise_confusion_mat(train_seg_confusion)
        train_dist_confusion_n = normalise_confusion_mat(train_dist_confusion)
        val_seg_confusion_n = normalise_confusion_mat(val_seg_confusion)
        val_dist_confusion_n = normalise_confusion_mat(val_dist_confusion)

        # Log confusion matrices to tensorboard
        log_confusion_mat(train_seg_confusion, (16,10), "train_confusion_seg", "0.0f", epoch)
        log_confusion_mat(train_dist_confusion, (9,7), "train_confusion_dist", "0.0f", epoch)
        log_confusion_mat(val_seg_confusion, (16,10), "val_confusion_seg", "0.0f", epoch)
        log_confusion_mat(val_dist_confusion, (9,7), "val_confusion_dist", "0.0f", epoch)

        # Log normalised confusion matrices to tensorboard
        log_confusion_mat(train_seg_confusion_n, (16,10), "train_confusion_seg_n", "0.3f", epoch)
        log_confusion_mat(train_dist_confusion_n, (9,7), "train_confusion_dist_n", "0.3f", epoch)
        log_confusion_mat(val_seg_confusion_n, (16,10), "val_confusion_seg_n", "0.3f", epoch)
        log_confusion_mat(val_dist_confusion_n, (9,7), "val_confusion_dist_n", "0.3f", epoch)        


    # Save checkpoint
    if epoch % 100 == 0 and epoch != 0:
        save_path = os.path.join(PRETRAINED_PATH, "%s_%s_%s_epoch%d.pth"
                    %(MODEL_NAME, MODEL_BACKBONE, DATA_NAME, epoch))
        torch.save(model.state_dict(), save_path)
        print("%s has been saved." %save_path)

tbX_logger.close()

# Save final trained model
save_path = os.path.join(PRETRAINED_PATH, "%s_%s_%s_epoch%d_final.pth"
            %(MODEL_NAME, MODEL_BACKBONE, DATA_NAME, TRAIN_EPOCHS))
torch.save(model.state_dict(), save_path)
print("FINISHED: %s has been saved." %save_path)

# list of TODO's:

# find out what layers[0] means for the backbone network

# try focal loss
# implement two-head model
# add intermediate vector stage (maybe not needed)
# implement coco dataset training