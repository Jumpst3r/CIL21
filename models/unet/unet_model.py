""" Full assembly of the parts to form the complete network """
# adapted from https://github.com/milesial/Pytorch-UNet

import torch.nn.functional as F
import pytorch_lightning as pl
from .utils import IoU, DiceBCELoss, IoULoss, FocalLoss, F1
from torchgeometry.losses.dice import DiceLoss
import numpy as np
from .unet_parts import *


class StackedUNet(pl.LightningModule):
    def __init__(self, lr=1e-3, nb_blocks=5):
        super(StackedUNet, self).__init__()
        self.initBlock = UNet(n_channels=3, return_features=True)
        
        self.blocks = nn.ModuleList(UNet(n_channels=64, return_features=True) for _ in range(nb_blocks-1))

        self.conv_up_input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, padding=0)
        self.conv_up_logits_init = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=1, padding=0)
        self.conv_up_logits = nn.ModuleList(nn.Conv2d(in_channels=1, out_channels=64, kernel_size=1, padding=0) for _ in range(nb_blocks-1))

        self.lr = lr
        self.training = True
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, gamma=0.5, step_size=100, verbose=True)
        self.alpha = 0.1
        self.loss = F.binary_cross_entropy_with_logits
        self.IoU = IoU
        self.testIoUs = []
        self.testF1s = []
    
    def forward(self, x):
        feat1, y1 = self.initBlock(x)
        p1 = self.conv_up_logits_init(torch.sigmoid(y1))
        yL = [y1]
        probL = [p1]
        featL = [feat1]

        in_stacked = self.conv_up_input(x)

        for conv_up_logits_init, block in zip(self.conv_up_logits, self.blocks):
            # input of the last stack + feature output of the last stack + probability prediction of the last stack
            in_stacked = in_stacked + featL[-1] + probL[-1]
            feat, y = block(in_stacked)
            yL.append(y)
            probL.append(conv_up_logits_init(y))
            featL.append(feat)

        if self.training:
            return yL
        else:
            return yL[-1]

    def training_step(self, batch, batch_idx):
        x, y = batch
        yL = self.forward(x)
        # loss = self.loss(yL[-1], y) + self.alpha * sum([self.loss(yhat, y) for yhat in yL[:-1]])
        loss = sum([self.loss(yhat, y) for yhat in yL])
        self.log('training loss', loss)
        self.log('lr', self.lr)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        loss = self.loss(out, y)
        iou = self.IoU(out,y)
        self.log('IoU val', iou)
        self.log('loss val', loss)
        return {'loss val':loss, 'IoU val': iou}

    def configure_optimizers(self):
        return {
            'optimizer': self.optimizer,
            'lr_scheduler': self.lr_scheduler,
            # 'monitor': 'training loss'
        }    

    def test_step(self, batch, batch_idx):
        x,y = batch
        out = self.forward(x)
        iou = self.IoU(out,y)
        f = F1(out,y)
        self.testIoUs.append(iou.detach().cpu().item())
        self.testF1s.append(f.detach().cpu().item())

    def test_epoch_end(self, outputs):
        IoUs = np.array(self.testIoUs).mean()
        f = np.array(self.testF1s).mean()
        logs = {'IoU': IoUs, 'results': (IoUs, f)}
        return {'results':(IoUs, f), 'F1':f,'progress_bar': logs}

class UNet(pl.LightningModule):
    def __init__(self, lr=1e-4, n_channels=3, n_classes=1, return_features=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.return_features = return_features
        self.inc = DoubleConv(self.n_channels, 64)
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
        if self.return_features:
            return x, logits
        else:
            return logits

   