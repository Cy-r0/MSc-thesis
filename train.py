from datetime import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
import torch.optim as optim
from torch.utils.data import DataLoader, sampler
import torchvision.transforms as T
from torchvision.utils import make_grid

from datasets.voc_distance import VOCDistance
from models.deeplabv3plus_Y import Deeplabv3plus_Y
from models.sync_batchnorm import DataParallelWithCallback
from models.sync_batchnorm.replicate import patch_replication_callback
import transforms.transforms as myT

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
DATALOADER_JOBS = 4
DSET_ROOT = "/home/cyrus/Datasets"

DATA_RESCALE = 512
DATA_RANDOMCROP = 512

TRAIN_GPUS = 2
TEST_GPUS = 1

RESUME = True
PRETRAINED_PATH = "models/pretrained"

BATCH_SIZE = 9
TRAIN_LR = 0.007
TRAIN_POWER = 0.9
TRAIN_MOMENTUM = 0.9
TRAIN_EPOCHS = 46
VAL_FRACTION = 0.2

MODEL_NAME = 'deeplabv3plus_Y'
MODEL_BACKBONE = 'xception'
DATA_NAME = 'VOC2012'


# Helper methods
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


# Dataset transforms
transform = T.Compose([myT.Quantise(level_widths=[1,2,2,3,3,4,5,6,8,10,14,20]),
                       myT.Resize(DATA_RESCALE),
                       myT.RandomCrop(DATA_RANDOMCROP),
                       myT.ToTensor(),
                       myT.Normalise((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Dataset setup
VOCW_train = VOCDistance(DSET_ROOT,
                         image_set="train_reduced",
                         transform=transform)
VOCW_val = VOCDistance(DSET_ROOT,
                       image_set="val_reduced",
                       transform=transform)

loader_train = DataLoader(VOCW_train,
                          batch_size = BATCH_SIZE,
                          sampler=sampler.SubsetRandomSampler(
                              range(int(len(VOCW_train) * (1 - VAL_FRACTION)))),
                          num_workers = DATALOADER_JOBS,
                          pin_memory = True,
                          drop_last=True)
loader_val = DataLoader(VOCW_train,
                        batch_size = BATCH_SIZE,
                        sampler=sampler.SubsetRandomSampler(
                            range(int(len(VOCW_train) * (1 - VAL_FRACTION)),
                                  len(VOCW_train))),
                        num_workers = DATALOADER_JOBS,
                        pin_memory = True,
                        drop_last=True)
loader_test = DataLoader(VOCW_val,
)


# show images TODO: delete this
"""
dataiter = iter(loader_train)
batch = dataiter.next()

img = batch["image"]
tgt = batch["target"]

show(make_grid(img))
plt.show()
show(make_grid(tgt))
plt.show()
"""

now = datetime.now()

# Initialise tensorboardX logger
if LOG:
    tbX_logger = SummaryWriter(os.path.join(LOG_PATH,
                                            now.strftime("%Y%m%d-%H%M")))

# Model setup
model = Deeplabv3plus_Y(n_classes=13)

# Move model to GPU devices
device = torch.device(0)
print("GPUs found:", torch.cuda.device_count())
print("GPUs used:", TRAIN_GPUS)

if TRAIN_GPUS > 1:
    model = nn.DataParallel(model)

# Load pretrained model weights only for backbone network and aspp
if RESUME:
    current_dict = model.state_dict()
    pretrained_dict = torch.load(os.path.join(PRETRAINED_PATH,
                        "deeplabv3plus_xception_VOC2012_epoch46_all.pth"))
    pretrained_dict = {k: v for k, v in pretrained_dict.items()
                       if "backbone" in k
                       or "aspp" in k}

    current_dict.update(pretrained_dict)
    model.load_state_dict(current_dict)

model.to(device)

# Set training parameters
criterion = nn.CrossEntropyLoss()#ignore_index=0)
optimiser = optim.SGD(
        params = [
            {'params': get_params(model, key='1x'), 'lr': TRAIN_LR},
            {'params': get_params(model, key='10x'), 'lr': 10*TRAIN_LR}
        ],
        momentum=TRAIN_MOMENTUM)

# Training loop
i = 0
max_i = TRAIN_EPOCHS * len(loader_train)
for epoch in range(TRAIN_EPOCHS):

    train_loss = 0.0
    
    model.train()
    for train_batch_i, train_batch in enumerate(loader_train):

        train_inputs = train_batch["image"].to(device)
        train_labels = train_batch["target"].to(device)
        # Convert labels to integers
        train_labels = train_labels.mul(255).round().long().squeeze()

        lr = adjust_lr(optimiser, i, max_i)
        optimiser.zero_grad()

        # Calculate loss
        train_predictions = model(train_inputs)
        loss = criterion(train_predictions, train_labels)

        # Backprop and gradient descent
        loss.backward()
        optimiser.step()

        train_loss += loss.item()
        i += 1


    val_loss = 0.0

    model.train()
    with torch.no_grad():
        for val_batch_i, val_batch in enumerate(loader_val):

            val_inputs = val_batch["image"].to(device)
            val_labels = val_batch["target"].to(device)
            val_labels = val_labels.mul(255).round().long().squeeze()

            val_predictions = model(val_inputs)
            loss = criterion(val_predictions, val_labels)

            val_loss += loss.item()


    # average losses on all batches
    train_loss /= len(loader_train)
    val_loss /= len(loader_val)

    print("epoch: %d/%d\ti: %d\tlr: %g\ttrain_loss: %g\tval_loss: %g"
          % (epoch+1, TRAIN_EPOCHS, i+1, lr, train_loss, val_loss))

    # Log stats on tensorboard
    # Only get first image of batch (TODO: change to whole batch)
    train_input_tb = make_grid(train_inputs).cpu().numpy()
    train_label_tb = make_grid(train_labels.unsqueeze(1)).cpu().numpy()

    val_input_tb = make_grid(val_inputs).cpu().numpy()
    val_label_tb = make_grid(val_labels.unsqueeze(1)).cpu().numpy()

    train_prediction_tb = torch.argmax(train_predictions[0],
                                       dim=0).cpu().unsqueeze(0).numpy()
    val_prediction_tb = torch.argmax(val_predictions[0],
                                     dim=0).cpu().unsqueeze(0).numpy()

    train_pix_accuracy = (np.sum(train_label_tb == train_prediction_tb)
                         / (DATA_RESCALE ** 2))
    val_pix_accuracy = (np.sum(val_label_tb == val_prediction_tb)
                         / (DATA_RESCALE ** 2))
    
    tbX_logger.add_scalars("losses", {"train": train_loss,
                                      "val": val_loss}, epoch)
    tbX_logger.add_scalars("pixel accuracies", {"train": train_pix_accuracy,
                                                "val": val_pix_accuracy}, epoch)
    tbX_logger.add_scalar("lr", lr, epoch)

    # it seems like tensorboard doesn't like saving a lot of images,
    # so save only every 10 epochs
    if epoch % 10 == 0:
        tbX_logger.add_image("train_input", train_input_tb, epoch)
        tbX_logger.add_image("train_label", train_label_tb, epoch)
        tbX_logger.add_image("train_prediction", train_prediction_tb, epoch)

    # Save checkpoint
    if epoch % 100 == 0 and epoch != 0:
        save_path = os.path.join(PRETRAINED_PATH, "%s_%s_%s_epoch%d.pth"
                    %(MODEL_NAME, MODEL_BACKBONE, DATA_NAME, epoch))
        torch.save(model.state_dict(), save_path)
        print("%s has been saved." %save_path)


# Save final trained model
save_path = os.path.join(PRETRAINED_PATH, "%s_%s_%s_epoch%d_final.pth"
            %(MODEL_NAME, MODEL_BACKBONE, DATA_NAME, TRAIN_EPOCHS))
torch.save(model.state_dict(), save_path)
print("FINISHED: %s has been saved." %save_path)

# list of TODO's:
# save targets and predictions as colours instead of class number
# make same images appear at multiple timesteps on tensorboardX
# fix eval() loss which is currently way higher than train() loss
#   leads: set track_runnin_stats to false; dont reuse same bn layer in multiple places

# implement two-head model
# add intermediate vector stage (maybe not needed)