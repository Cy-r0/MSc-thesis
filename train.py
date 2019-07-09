import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T

from datasets.voc_watershed import VOCWatershed

# Constants here
BATCH_SIZE = 10
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
DATALOADER_JOBS = 0
DSET_ROOT = "/home/cyrus/Datasets"


transform = T.ToTensor()

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
                          pin_memory = True)
loader_val = DataLoader(VOCW_val,
                        batch_size = BATCH_SIZE,
                        shuffle = False,
                        num_workers = DATALOADER_JOBS,
                        pin_memory = True)



