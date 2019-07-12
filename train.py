from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T

from datasets.voc_watershed import VOCWatershed
from models.deeplabv3plus_Y import Deeplabv3plus_Y
from models.sync_batchnorm.replicate import patch_replication_callback

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

RESUME = False
PRETRAINED_PATH = """models/pretrained/
                    deeplabv3plus_xception_VOC2012_itr46_all.pth"""

BATCH_SIZE = 16
TRAIN_LR = 0.007
TRAIN_POWER = 0.9
TRAIN_MOMENTUM = 0.9
TRAIN_EPOCHS = 46


MODEL_NAME = 'deeplabv3plus'
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


# Dataset transforms

# TODO: implement quantisatin transform

transform = T.Compose([T.Normalize(0,)
                       T.Resize(DATA_RESCALE),
                       T.RandomCrop(DATA_RANDOMCROP),
                       T.ToTensor()])

# Dataset setup
VOCW_train = VOCWatershed(DSET_ROOT,
                          image_set = "train",
                          transform = transform,
                          target_transform = transform)
VOCW_val = VOCWatershed(DSET_ROOT,
                        image_set = "val",
                        transform = transform,
                        target_transform = transform)

loader_train = DataLoader(VOCW_train,
                          batch_size = BATCH_SIZE,
                          shuffle = False,
                          num_workers = DATALOADER_JOBS,
                          pin_memory = True,
                          drop_last=True)
loader_val = DataLoader(VOCW_val,
                        batch_size = BATCH_SIZE,
                        shuffle = False,
                        num_workers = DATALOADER_JOBS,
                        pin_memory = True,
                        drop_last=True)

# Initialise tensorboardX logger
if LOG:
    tbX_logger = SummaryWriter(LOG_PATH)

# Model setup
model = Deeplabv3plus_Y()

# Move model to GPU devices
device = torch.device(0)
print("Number of GPUs:", torch.cuda.device_count())
if TRAIN_GPUS > 1:
    model = nn.DataParallel(model)
    patch_replication_callback(model)
model.to(device)

# Load pretrained model weights and start training from that point
if RESUME:
    current_dict = model.state_dict()
    pretrained_dict = torch.load(PRETRAINED_PATH)

    current_dict.update(pretrained_dict)
    model.load_state_dict(current_dict)

# Set training parameters
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimiser = optim.SGD(
        params = [
            {'params': get_params(model.module, key='1x'), 'lr': TRAIN_LR},
            {'params': get_params(model.module, key='10x'), 'lr': 10*TRAIN_LR}
        ],
        momentum=TRAIN_MOMENTUM)

# Training loop
i = 0
max_i = TRAIN_EPOCHS * len(loader_train)
for epoch in range(TRAIN_EPOCHS):

    running_loss = 0.0
    for batch_i, batch in enumerate(loader_train):

        inputs, labels = batch["image"], batch["target"].to(device[0])

        lr = adjust_lr(optimiser, i, max_i)
        optimiser.zero_grad()
        # Calculate loss
        predictions = model(inputs).to(device[0])
        loss = criterion(predictions, labels)

        # Backprop and gradient descent
        loss.backward()
        optimiser.step()

        running_loss += loss.item()

        print("epoch: %d/%d\tbatch: %d/%d\titr: %d\tlr: %g\tloss: %g"
              % (epoch, TRAIN_EPOCHS, batch_i,
                 VOCW_train.__len__() // BATCH_SIZE, i, lr, running_loss))

        # Log stats on tensorboard
        if i % 100 == 0:
            input_tb = inputs.numpy()[0]
            label_tb = labels[0].cpu().numpy()
            prediction_tb = torch.argmax(predictions[0], dim=0).cpu().numpy()
            pix_accuracy = np.sum(label_tb == prediction_tb)
                           // (DATA_RESCALE ** 2)
            
            tbX_logger.add_scalar("loss", running_loss, i)
            tbX_logger.add_scalar("lr", lr, i)
            tbX_logger.add_scalar("pixel_accuracy", pix_accuracy, i)
            tbX_logger.add_image("input", input_tb, i)
            tbX_logger.add_image("label", label_tb, i)
            tbX_logger.add_image("prediction", prediction_tb, i)
        
        running_loss = 0.0

        # Save checkpoint
        if i % 5000 == 0:
            save_path = os.path.join(PRETRAINED_PATH, "%s_%s_%s_itr%d.pth"
                        %(MODEL_NAME, MODEL_BACKBONE, DATA_NAME, i))
			torch.save(model.state_dict(), save_path)
			print("%s has been saved" %save_path)

        i += 1

# Save final trained model
save_path = os.path.join(PRETRAINED_PATH, "%s_%s_%s_epoch%d_all.pth"
            %(MODEL_NAME, MODEL_BACKBONE, DATA_NAME, TRAIN_EPOCHS))
torch.save(model.state_dict(), save_path)
print("Final: %s has been saved" %save_path)