import torch

class VOCSettings(object):
    """
    Data container for training and testing the model on the VOC dataset.
    """

    def __init__(self):

        self.LOG = True
        self.LOG_PATH = "logs"

        # Model settings
        self.RESUME = True
        self.PRETRAINED_PATH = "models/pretrained"
        self.MODEL_NAME = "deeplabv3plus_multitask"
        self.MODEL_BACKBONE = "xception"
        self.DATA_NAME = "VOC2012"


        # Dataset settings
        self.DSET_ROOT = "/home/cyrus/Datasets"  
        self.DATALOADER_JOBS = 4

        self.DATA_RESCALE = 512
        self.DATA_RANDOMCROP = 512

        self.CLASSES = {0: "background",
                        1: "aeroplane",
                        2: "bicycle",
                        3: "bird",
                        4: "boat",
                        5: "bottle",
                        6: "bus",
                        7: "car",
                        8: "cat",
                        9: "chair",
                        10: "cow",
                        11: "diningtable",
                        12: "dog",
                        13: "horse",
                        14: "motorbike",
                        15: "person",
                        16: "pottedplant",
                        17: "sheep",
                        18: "sofa",
                        19: "train",
                        20: "tvmonitor",
                        255: "unlabelled"}
        self.N_CLASSES = len(self.CLASSES)

        self.LEVEL_WIDTHS = [1,5,6,8,9,10,12,14,20]
        self.N_ENERGY_LEVELS = len(self.LEVEL_WIDTHS) + 1


        # Training settings
        self.TRAIN_GPUS = 2
        self.TRAIN_BATCH_SIZE = 8
        self.TRAIN_EPOCHS = 45
        self.TRAIN_LR = 0.007
        self.TRAIN_MOMENTUM = 0.9
        self.TRAIN_POWER = 0.9

        self.ADJUST_LR = True

        self.IMG_LOG_EPOCHS = 10


        # Validation settings
        self.VAL_FRACTION = 0.5
        self.VAL_GPUS = 2
        self.VAL_BATCH_SIZE = 8