""" Full assembly of the parts to form the complete network """
# adapted from https://github.com/milesial/Pytorch-UNet

import torch.nn.functional as F
import pytorch_lightning as pl
from .utils import IoU, dice_loss

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
        weights = torch.ones(size=y.shape, dtype=torch.float32).to(self.device)
        total = torch.sum((y==1) | (y==0), dim=(1,2,3))
        pos_samples = torch.sum((y==1), dim=(1,2,3))
        neg_samples = torch.sum((y==0), dim=(1,2,3))
        try:
            weights[:,y[0]==0] =  (pos_samples / total).unsqueeze(1)
            weights[:,y[1]==1] =  (neg_samples / total).unsqueeze(1)
            loss = F.binary_cross_entropy_with_logits(out, y, weight=weights)
        except IndexError:
            loss = F.binary_cross_entropy_with_logits(out, y)
        self.log('training loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        weights = torch.ones(size=y.shape, dtype=torch.float32).to(self.device)
        total = torch.sum((y==1) | (y==0), dim=(1,2,3))
        pos_samples = torch.sum((y==1), dim=(1,2,3))
        neg_samples = torch.sum((y==0), dim=(1,2,3))
        try:
            weights[:,y[0]==0] =  (pos_samples / total).unsqueeze(1)
            weights[:,y[1]==1] =  (neg_samples / total).unsqueeze(1)
            loss = F.binary_cross_entropy_with_logits(out, y, weight=weights)
        except IndexError:
            loss = F.binary_cross_entropy_with_logits(out, y)

        iou = IoU(y,out)
        self.log('IoU val', iou)
        self.log('loss val', loss)
        return {'loss val':loss, 'IoU val': iou}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        #lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        #scheduler = {'scheduler': lr_scheduler, 'interval': 'step', 'monitor': 'training loss'}

        return optimizer