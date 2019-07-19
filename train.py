from datetime import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
import torch.optim as optim
from torch.utils.data import DataLoader
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
                          shuffle = True,
                          num_workers = DATALOADER_JOBS,
                          pin_memory = True,
                          drop_last=True)
loader_val = DataLoader(VOCW_val,
                        batch_size = BATCH_SIZE,
                        shuffle = True,
                        num_workers = DATALOADER_JOBS,
                        pin_memory = True,
                        drop_last=True)

# show images TODO: delete this

dataiter = iter(loader_train)
batch = dataiter.next()

img = batch["image"]
tgt = batch["target"]

show(make_grid(img))
plt.show()
show(make_grid(tgt))
plt.show()


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

    running_loss = 0.0
    for batch_i, batch in enumerate(loader_train):

        # Training mode
        model.train()

        inputs, labels = batch["image"].to(device), batch["target"].to(device)
        # Convert labels to integers
        labels = labels.mul(255).round().long().squeeze()

        lr = adjust_lr(optimiser, i, max_i)
        optimiser.zero_grad()

        # Calculate loss
        predictions = model(inputs)
        loss = criterion(predictions, labels)

        # Backprop and gradient descent
        loss.backward()
        optimiser.step()

        running_loss += loss.item()

        # Evaluation mode
        #model.eval()

        # Predict on validation data
        #val_predictions = 

        print("epoch: %d/%d\tbatch: %d/%d\titr: %d\tlr: %g\tloss: %g"
              % (epoch+1, TRAIN_EPOCHS, batch_i+1,
                 VOCW_train.__len__() // BATCH_SIZE, i+1, lr, running_loss))

        # Log stats on tensorboard
        if i % 100 == 0:
            # Only get first image of batch (TODO: change to whole batch)
            input_tb = inputs.cpu().numpy()[0]
            label_tb = labels.cpu()[0].unsqueeze(0).numpy()

            prediction_tb = torch.argmax(predictions[0], dim=0).cpu().unsqueeze(0).numpy()
            pix_accuracy = (np.sum(label_tb == prediction_tb)
                            // (DATA_RESCALE ** 2))
            
            tbX_logger.add_scalar("train_loss", running_loss, i)
            #tbX_logger.add_scalar("val_loss", val_loss, i)

            tbX_logger.add_scalar("lr", lr, i)

            tbX_logger.add_scalar("pixel_accuracy", pix_accuracy, i)

            tbX_logger.add_image("input", input_tb, i)
            tbX_logger.add_image("label", label_tb, i)
            tbX_logger.add_image("prediction", prediction_tb, i)
        
        running_loss = 0.0

        # Save checkpoint
        if i % 5000 == 0 and i != 0:
            save_path = os.path.join(PRETRAINED_PATH, "%s_%s_%s_itr%d.pth"
                        %(MODEL_NAME, MODEL_BACKBONE, DATA_NAME, i))
            torch.save(model.state_dict(), save_path)
            print("%s has been saved" %save_path)

        i += 1

# Save final trained model
save_path = os.path.join(PRETRAINED_PATH, "%s_%s_%s_epoch%d_final.pth"
            %(MODEL_NAME, MODEL_BACKBONE, DATA_NAME, TRAIN_EPOCHS))
torch.save(model.state_dict(), save_path)
print("Final: %s has been saved" %save_path)

# list of TODO's:
# split training dataset to get a validation set
# implement validation loss
# save targets and predictions as colours instead of class number