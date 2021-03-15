""" Full assembly of the parts to form the complete network """
# adapted from https://github.com/milesial/Pytorch-UNet

import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.ops.focal_loss import sigmoid_focal_loss
from .utils import IoU

from .unet_parts import *


class UNet(pl.LightningModule):
    def __init__(self):
        super(UNet, self).__init__()
        self.n_channels = 3
        self.n_classes = 1

        self.inc = DoubleConv(3, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        loss = F.binary_cross_entropy_with_logits(out, y)
        self.log('training loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        loss = F.binary_cross_entropy_with_logits(out, y)
        iou = IoU(y,out)
        self.log('IoU val', iou)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer