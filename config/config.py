import torch

class VOCConfig(object):
    """
    Data container for training and testing the model on the VOC dataset.
    """

    def __init__(self):

        # LOG SETTINGS

        self.LOG = True
        self.LOG_PATH = "logs"


        # MODEL SETTINGS

        self.USE_PRETRAINED = True
        self.PRETRAINED_PATH = "models/pretrained"
        self.TRAINED_PATH = "models/pretrained"
        self.MODEL_NAME = "deeplabv3plus_multitask"
        self.MODEL_BACKBONE = "xception"
        self.DATA_NAME = "VOC2012"


        # DATASET SETTINGS

        self.DSET_ROOT = "/home/cyrus/Datasets"  
        self.DATALOADER_JOBS = 8
        self.CLASSES = {
            0: "background",
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
            20: "tvmonitor"
        }
        self.N_CLASSES = len(self.CLASSES)
        # NOTE: I dont recommend setting the width of the lowest level to 1 pixel,
        # because findcontours() can leak through it
        self.LEVEL_WIDTHS = [1,1,2,6,8,12,15,25,39]
        self.N_ENERGY_LEVELS = len(self.LEVEL_WIDTHS) + 1


        # DATA AUGMENTATION SETTINGS

        self.DATA_RESCALE = 512
        self.RESIZEDCROP_SCALE_RANGE = (0.5, 1.)
        # The colour parameters below are maximum deviations allowed from
        # the original image colour
        self.BRIGHTNESS = 0.5
        self.CONTRAST = 0.3
        self.SATURATION = 0.5
        self.HUE = 0.1


        # TRAINING SETTINGS

        self.TRAIN_GPUS = 2
        # IMPORTANT! If you use DistributedDataParallel,
        # TRAIN_BATCH_SIZE is the batch size contained in each GPU
        # so the total batch size will be BATCH_SIZE * GPUS
        self.TRAIN_BATCH_SIZE = 4
        self.TRAIN_EPOCHS = 40
        self.TRAIN_LR = 0.005
        self.TRAIN_MOMENTUM = 0.9
        self.TRAIN_POWER = 0.9

        self.ADJUST_LR = True

        self.IMG_LOG_EPOCHS = 10


        # VALIDATION SETTINGS

        self.VAL_FRACTION = 0.5
        self.VAL_BATCH_SIZE = self.TRAIN_BATCH_SIZE


        # TEST SETTINGS

        self.TEST_GPUS = 1
        self.TEST_BATCH_SIZE = 1