# ----------------------------------------
# Written by Yude Wang
# Modified by Ciro Cursio
# ----------------------------------------

from timeit import default_timer as timer

import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from models.ASPP import ASPP
# from models.sync_batchnorm import nn.SyncronizedBatchNorm2d
from models.xception import Xception

ASPP_OUT_DIM = 256

OUTPUT_STRIDE = 16
IN_DIM = 256
SHORTCUT_DIM = 48
SHORTCUT_KERNEL = 1

BN_MOMENTUM = 0.0003

class Deeplabv3plus_Y(nn.Module):
    """
    Neural network based on DeepLab v3+ but with two branches, one for semantic
    segmentation and the other for watershed-based instance segmentation.
    """

    def __init__(self, n_classes):
        super(Deeplabv3plus_Y, self).__init__()

        input_channel = 2048

        self.backbone = Xception(os=OUTPUT_STRIDE)
        self.aspp = ASPP(dim_in=input_channel, 
                         dim_out=ASPP_OUT_DIM, 
                         rate=16//OUTPUT_STRIDE,
                         bn_mom=BN_MOMENTUM)
        self.dropout1 = nn.Dropout(0.5)
        self.upsample4 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.upsample_sub = nn.UpsamplingBilinear2d(scale_factor=OUTPUT_STRIDE//4)

        self.shortcut_conv = nn.Sequential(
                nn.Conv2d(IN_DIM,
                          SHORTCUT_DIM,
                          SHORTCUT_KERNEL,
                          1,
                          padding=SHORTCUT_KERNEL//2,
                          bias=True),
                nn.BatchNorm2d(SHORTCUT_DIM, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),        
        )        
        self.cat_conv = nn.Sequential(
                nn.Conv2d(ASPP_OUT_DIM + SHORTCUT_DIM,
                ASPP_OUT_DIM, 3, 1, padding=1, bias=True),
                nn.BatchNorm2d(ASPP_OUT_DIM, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Conv2d(ASPP_OUT_DIM, ASPP_OUT_DIM, 3, 1, padding=1, bias=True),
                nn.BatchNorm2d(ASPP_OUT_DIM, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
        )
        self.cls_conv = nn.Conv2d(ASPP_OUT_DIM, n_classes, 1, 1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # If model started training set all batchnorm tracking
        if self.training:
            _set_track_running_stats(self, val=True)
        # If model changed to eval mode reset tracking
        elif not self.training:
            _set_track_running_stats(self, val=False)
        
        _ = self.backbone(x)
        layers = self.backbone.get_layers()

        feature_aspp = self.aspp(layers[-1])
        feature_aspp = self.dropout1(feature_aspp)
        feature_aspp = self.upsample_sub(feature_aspp)

        feature_shallow = self.shortcut_conv(layers[0])
        feature_cat = torch.cat([feature_aspp,feature_shallow], 1)
        result = self.cat_conv(feature_cat) 
        result = self.cls_conv(result)
        result = self.upsample4(result)

        return result


def _set_track_running_stats(model, val):
    for m in model.children():

        # if layer has children, go down one level
        if list(m.children()):
            _set_track_running_stats(m, val)

        # if layer has no children and its batchnorm, set value
        else:
            if isinstance(m, nn.BatchNorm2d):
                m.track_running_stats = val