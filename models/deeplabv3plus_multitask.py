from timeit import default_timer as timer

import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from models.ASPP import ASPP
from models.xception import Xception


# Model parameters
IN_CH = 256

ASPP_IN_CH = 2048
ASPP_OUT_CH = 256

SHORTCUT_CH = 48
SHORTCUT_KERNEL = 1

OUTPUT_STRIDE = 16

BN_MOMENTUM = 0.3


class Deeplabv3plus_multitask(nn.Module):
    """
    Neural network based on DeepLab v3+ but with two branches, one for semantic
    segmentation and the other for watershed-based instance segmentation.
    """

    def __init__(self, seg_classes, dist_classes):
        super(Deeplabv3plus_multitask, self).__init__()

        self.backbone = Xception(
            output_stride=OUTPUT_STRIDE,
            bn_momentum=BN_MOMENTUM)
        self.aspp = ASPP(
            dim_in=ASPP_IN_CH, 
            dim_out=ASPP_OUT_CH, 
            rate=16 // OUTPUT_STRIDE,
            bn_momentum=BN_MOMENTUM)
        self.dropout1 = nn.Dropout(0.5)
        self.upsample_sub = nn.UpsamplingBilinear2d(scale_factor=OUTPUT_STRIDE//4)

        # First branch
        self.shortcut_conv_1 = nn.Sequential(
            nn.Conv2d(
                IN_CH,
                SHORTCUT_CH,
                SHORTCUT_KERNEL,
                1,
                padding=SHORTCUT_KERNEL // 2,
                bias=True),
            nn.BatchNorm2d(SHORTCUT_CH, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True))        
        self.cat_conv_1 = nn.Sequential(
            nn.Conv2d(ASPP_OUT_CH + SHORTCUT_CH,
            ASPP_OUT_CH, 3, 1, padding=1, bias=True),
            nn.BatchNorm2d(ASPP_OUT_CH, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(ASPP_OUT_CH, ASPP_OUT_CH, 3, 1, padding=1, bias=True),
            nn.BatchNorm2d(ASPP_OUT_CH, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1))
        self.cls_conv_1 = nn.Conv2d(ASPP_OUT_CH, seg_classes, 1, 1, padding=0)

        # Second branch
        self.shortcut_conv_2 = nn.Sequential(
            nn.Conv2d(
                IN_CH,
                SHORTCUT_CH,
                SHORTCUT_KERNEL,
                1,
                padding=SHORTCUT_KERNEL//2,
                bias=True),
            nn.BatchNorm2d(SHORTCUT_CH, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True))
        self.cat_conv_2 = nn.Sequential(
            nn.Conv2d(ASPP_OUT_CH + SHORTCUT_CH,
            ASPP_OUT_CH, 3, 1, padding=1, bias=True),
            nn.BatchNorm2d(ASPP_OUT_CH, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(ASPP_OUT_CH, ASPP_OUT_CH, 3, 1, padding=1, bias=True),
            nn.BatchNorm2d(ASPP_OUT_CH, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1))
        self.cls_conv_2 = nn.Conv2d(ASPP_OUT_CH, dist_classes, 1, 1, padding=0)

        # Weight initialisation
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        
        self.upsample4_approx = nn.UpsamplingBilinear2d(
            size=(x.shape[2], x.shape[3]))

        _ = self.backbone(x)
        layers = self.backbone.get_layers()

        feature_aspp = self.aspp(layers[-1])
        feature_aspp = self.dropout1(feature_aspp)
        feature_aspp = self.upsample_sub(feature_aspp)

        self.upsample_slightly = nn.UpsamplingBilinear2d(
            size=(feature_aspp.shape[2], feature_aspp.shape[3]))

        # First branch
        feature_shallow_1 = self.shortcut_conv_1(layers[0])
        feature_shallow_1 = self.upsample_slightly(feature_shallow_1)
        feature_cat_1 = torch.cat([feature_aspp, feature_shallow_1], 1)
        result_1 = self.cat_conv_1(feature_cat_1)
        result_1 = self.cls_conv_1(result_1)
        result_1 = self.upsample4_approx(result_1)

        # Second branch
        feature_shallow_2 = self.shortcut_conv_2(layers[0])
        feature_shallow_2 = self.upsample_slightly(feature_shallow_2)
        feature_cat_2 = torch.cat([feature_aspp, feature_shallow_2], 1)
        result_2 = self.cat_conv_2(feature_cat_2)
        result_2 = self.cls_conv_2(result_2)
        result_2 = self.upsample4_approx(result_2)

        return result_1, result_2