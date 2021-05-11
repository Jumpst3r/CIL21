import pytorch_lightning as pl
import torch
import torch.nn as nn
from helpers import get_accuracy
import cv2

def conv_block(in_c, out_c, k=3, p=1):
    mid_c = out_c
    return nn.Sequential(
        nn.Conv2d(in_channels=in_c, out_channels=mid_c, kernel_size=k, padding=p),
        nn.BatchNorm2d(mid_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=mid_c, out_channels=out_c, kernel_size=k, padding=p),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


def upconv_block(in_c, out_c, k, s):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels=in_c, out_channels=out_c, kernel_size=k, stride=s),
        nn.BatchNorm2d(out_c)
    )


def downconv_block(in_c, out_c, k=3, d=2, p=1):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=k, padding=p, dilation=d),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True)
    )