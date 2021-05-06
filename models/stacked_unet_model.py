import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet_model import UNet


class StackedUNet(nn.Module):
    def __init__(self, n_channels, n_classes=1, n_stacks=4, bilinear=True, norm='batch'):
        super(StackedUNet, self).__init__()

        self.n_stacks = n_stacks
        n_channels_per_stack = [n_channels] + [64] * (n_stacks - 1)

        # UNet blocks
        self.unets = nn.ModuleList([
            nn.Sequential(
                UNet(n_channels_per_stack[i], n_classes, bilinear, norm),
            ) for i in range(n_stacks)])

        # to bring prediction to 64 channels
        self.conv_cup_1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=1, padding=0)
        self.conv_cup_2 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, padding=0)

    def forward(self, x):

        logits_list = []

        for i in range(self.n_stacks):
            if i == 0:
                unet_input = x
            else:
                logits64 = self.conv_cup_1(logits)
                if unet_input.shape[1] == 3:
                    unet_input = self.conv_cup_2(unet_input)
                unet_input = unet_input + features + logits64
            features, logits = self.unets[i](unet_input)
            logits_list.append(logits.clone())

        return logits_list
