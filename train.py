from datetime import datetime
import os
from pprint import pprint
from timeit import default_timer

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.metrics import confusion_matrix
from tensorboardX import SummaryWriter
from tensorboardX.utils import figure_to_image
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
import torch.optim as optim
from torch.utils.data import DataLoader, sampler
import torchvision.transforms as T
from torchvision.utils import make_grid

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
SEG_CLASSES = 21
DATALOADER_JOBS = 4
DSET_ROOT = "/home/cyrus/Datasets"

DATA_RESCALE = 512
DATA_RANDOMCROP = 512

TRAIN_GPUS = 2
TEST_GPUS = 2

RESUME = False
PRETRAINED_PATH = "models/pretrained"

TRAIN_BATCH_SIZE = 9
TEST_BATCH_SIZE = 9
TRAIN_LR = 0.007
TRAIN_POWER = 0.9
TRAIN_MOMENTUM = 0.9
TRAIN_EPOCHS = 45
VAL_FRACTION = 0.5
ADJUST_LR = False
LEVEL_WIDTHS = [1,5,6,8,9,10,12,14,20]
ENERGY_LEVELS = len(LEVEL_WIDTHS) + 1

MODEL_NAME = "deeplabv3plus_multitask"
MODEL_BACKBONE = 'xception'
DATA_NAME = 'VOC2012'


# Helper methods
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

def adjust_lr(optimizer, i, max_i):
    """
    Gradually decrease learning rate as iterations increase.
    """
    lr = TRAIN_LR * (1 - i/(max_i + 1)) ** TRAIN_POWER
    optimizer.param_groups[0]['lr'] = lr
    optimizer.param_groups[1]['lr'] = 10 * lr
    return lr

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
    
    model.train()
    for train_batch_i, train_batch in enumerate(loader_train):

        train_inputs = train_batch["image"].to(device)
        # Labels need to be converted from float 0-1 to integers
        train_seg = train_batch["seg"].to(device).mul(255).round().long().squeeze()
        train_dist = train_batch["dist"].to(device).mul(255).round().long().squeeze()

        if ADJUST_LR:
            lr = adjust_lr(optimiser, i, max_i)
        else:
            lr = TRAIN_LR

        optimiser.zero_grad()

        train_predicted_seg, train_predicted_dist = model(train_inputs)

        seg_loss = seg_criterion(train_predicted_seg, train_seg)
        dist_loss = dist_criterion(train_predicted_dist, train_dist)
        loss = seg_loss + dist_loss

        # Backprop and gradient descent
        loss.backward()
        optimiser.step()

        train_loss += loss.item()
        i += 1

        # Accumulate pixel belonging to each class (for weighted loss)
        #for class_i in range(ENERGY_LEVELS):
         #   counts[class_i] += torch.nonzero(train_labels.flatten() == class_i).flatten().size(0)

    # Print counts
    #counts = [c / (len(loader_train) * TRAIN_BATCH_SIZE) for c in counts]
    #print("Class counts per image:")
    #pprint(counts)


    val_loss = 0.0
    total_labels = torch.tensor((), dtype=torch.long)
    total_predictions = torch.tensor((), dtype=torch.long)

    model.eval()
    with torch.no_grad():
        for val_batch_i, val_batch in enumerate(loader_val):

            val_inputs = val_batch["image"].to(device)
            val_labels = val_batch["target"].to(device)
            val_labels = val_labels.mul(255).round().long().squeeze()


            #start.record()
            val_predictions = model(val_inputs)
            #end.record()
            #torch.cuda.synchronize()
            #print("forward time:", start.elapsed_time(end))

            loss = criterion(val_predictions, val_labels)
            val_loss += loss.item()

            # Accumulate labels and predictions for confusion matrix
            total_labels = torch.cat((total_labels, val_labels.cpu().flatten()))
            total_predictions = torch.cat((total_predictions,
                                           torch.argmax(val_predictions, dim=1).cpu().flatten()))


    # average losses on all batches
    train_loss /= len(loader_train)
    val_loss /= len(loader_val)

    print("epoch: %d/%d\ti: %d\tlr: %g\ttrain_loss: %g\tval_loss: %g"
          % (epoch+1, TRAIN_EPOCHS, i+1, lr, train_loss, val_loss))

    # Log stats on tensorboard
    train_input_tb = make_grid(train_inputs).cpu().numpy()
    val_input_tb = make_grid(val_inputs).cpu().numpy()

    train_label_tb = make_grid(colormap(train_labels.float().div(ENERGY_LEVELS)
                                        .unsqueeze(1).cpu())).numpy()
    val_label_tb = make_grid(colormap(val_labels.float().div(ENERGY_LEVELS)
                                      .unsqueeze(1).cpu())).numpy()

    train_prediction_tb = make_grid(colormap(torch.argmax(train_predictions, dim=1)
                                              .float().div(ENERGY_LEVELS).unsqueeze(1)
                                              .cpu())).numpy()
    val_prediction_tb = make_grid(colormap(torch.argmax(val_predictions, dim=1)
                                            .float().div(ENERGY_LEVELS).unsqueeze(1)
                                            .cpu())).numpy()

    train_pix_accuracy = (np.sum(train_label_tb == train_prediction_tb)
                         / (DATA_RESCALE ** 2))
    val_pix_accuracy = (np.sum(val_label_tb == val_prediction_tb)
                         / (DATA_RESCALE ** 2))
    
    tbX_logger.add_scalars("losses", {"train": train_loss,
                                      "val": val_loss}, epoch)
    tbX_logger.add_scalars("accuracies", {"train": train_pix_accuracy,
                                          "val": val_pix_accuracy}, epoch)
    tbX_logger.add_scalar("lr", lr, epoch)

    # it seems like tensorboard doesn't like saving a lot of images,
    # so save only every 10 epochs
    if epoch % 10 == 0:

        tbX_logger.add_image("train_input", train_input_tb, epoch)
        tbX_logger.add_image("train_label", train_label_tb, epoch)
        tbX_logger.add_image("train_prediction", train_prediction_tb, epoch)

        tbX_logger.add_image("val_input", val_input_tb, epoch)
        tbX_logger.add_image("val_label", val_label_tb, epoch)
        tbX_logger.add_image("val_prediction", val_prediction_tb, epoch)

        # Get confusion matrix
        confusion = confusion_matrix(total_labels, total_predictions).astype(float)
        # create normalised matrix
        confusion_n = np.copy(confusion)
        for i in range(len(confusion_n)):
            confusion_n[i] = confusion_n[i] / confusion_n[i].sum()

        # make figures, convert to images and log to tensorboard
        fig = plt.figure(figsize=(9,7))
        sn.heatmap(pd.DataFrame(confusion / (len(loader_val) * TEST_BATCH_SIZE)), annot=True, fmt=".0f")
        plt.ylabel("True")
        plt.xlabel("Predicted")
        confusion_img = figure_to_image(fig, close=True)
        tbX_logger.add_image("val_confusion_matrix", confusion_img, epoch)

        fig_n = plt.figure(figsize=(9,7))
        sn.heatmap(pd.DataFrame(confusion_n), annot=True)
        plt.ylabel("True")
        plt.xlabel("Predicted")
        confusion_n_img = figure_to_image(fig_n, close=True)
        tbX_logger.add_image("val_confusion_matrix_normalised", confusion_n_img, epoch)

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