""" Full assembly of the parts to form the complete network """
# adapted from https://github.com/milesial/Pytorch-UNet

import torch.nn.functional as F
import pytorch_lightning as pl
from .utils import IoU, DiceBCELoss, IoULoss, FocalLoss, F1, TverskyLoss, accuracy
# from torchgeometry.losses.dice import DiceLoss
import numpy as np
from .unet_parts import *
import torchvision
from .seg_net import SegNet
from.duc_hdc import ResNetDUCHDC, ResNetDUC


class StackedUNet(pl.LightningModule):
    def __init__(self, lr=1e-4, nb_blocks=4, share_weights=False, unet_mode='classic-backbone'):
        super(StackedUNet, self).__init__()

        if unet_mode == 'resnetduchdc' or unet_mode == 'resnetduc':
            self.initBlock = UNet(n_channels=3, return_features=False, mode=unet_mode)
            self.blocks = nn.ModuleList(
                UNet(n_channels=4, return_features=False, mode=unet_mode) for _ in range(nb_blocks - 1))

        else:
            if 'classic' in unet_mode or unet_mode == 'segnet':
                enc_dim = 64
            elif unet_mode == 'deeplab':
                enc_dim = 256
            else:
                enc_dim = 8
            self.initBlock = UNet(n_channels=3, return_features=True, mode=unet_mode)
            self.blocks = nn.ModuleList(UNet(n_channels=enc_dim, return_features=True, mode=unet_mode) for _ in range(nb_blocks-1))
            self.conv_up_input = nn.Conv2d(in_channels=3, out_channels=enc_dim, kernel_size=1, padding=0)
            self.conv_up_logits_init = nn.Conv2d(in_channels=1, out_channels=enc_dim, kernel_size=1, padding=0)
            self.conv_up_logits = nn.ModuleList(nn.Conv2d(in_channels=1, out_channels=enc_dim, kernel_size=1, padding=0) for _ in range(nb_blocks-1))

            if 'backbone' in unet_mode:
                rs50 = torchvision.models.resnet50(pretrained=True)
                self.backbone = torch.nn.Sequential(rs50.conv1, rs50.bn1, rs50.relu, rs50.maxpool, rs50.layer1, rs50.layer2,
                                                    rs50.layer3)

        self.unet_mode = unet_mode
        self.nb_blocks = nb_blocks
        self.lr = lr
        self.training = True
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5, amsgrad=True)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, gamma=0.5, step_size=100, verbose=True)
        self.alpha = 0.5
        self.loss = F.binary_cross_entropy_with_logits
        # self.loss = TverskyLoss()
        self.IoU = IoU
        self.testIoUs = []
        self.testF1s = []
        self.testaccs = []

    def forward(self, x):
        if self.unet_mode == 'resnetduchdc' or self.unet_mode == 'resnetduc':
            y1 = self.initBlock(x)
            yL = [y1]
            for block in self.blocks:
                in_stacked = torch.cat((x, yL[-1]), 1)
                y = block(in_stacked)
                yL.append(y)
        else:
            if 'backbone' in self.unet_mode:
                backbone_feats = self.backbone(x)
                feat1, y1 = self.initBlock(x, backbone_feats)
            else:
                feat1, y1 = self.initBlock(x)
            p1 = self.conv_up_logits_init(torch.sigmoid(y1))
            yL = [y1]
            probL = [p1]
            featL = [feat1]
            in_stacked = self.conv_up_input(x)
            for conv_up_logits_init, block in zip(self.conv_up_logits, self.blocks):
                # input of the last stack + feature output of the last stack + probability prediction of the last stack
                in_stacked = in_stacked + featL[-1] + probL[-1]
                if 'backbone-all' in self.unet_mode:
                    feat, y = block(in_stacked, backbone_feats)
                else:
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
        if self.nb_blocks > 1:
            loss = (1 - self.alpha) * self.loss(yL[-1], y) + self.alpha * sum([self.loss(yhat, y) for yhat in yL[:-1]]) / (self.nb_blocks - 1)
        else:
            loss = self.loss(yL[-1], y)
        # loss = sum([self.loss(yhat, y) for yhat in yL])
        self.log('training loss', loss)
        self.log('lr', self.lr)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        loss = self.loss(out, y)
        iou = self.IoU(out,y)
        f1 = F1(out,y)
        self.log('IoU val', iou)
        self.log('F1 val', f1)

        self.log('loss val', loss)
        return {'loss val': loss, 'IoU val': iou}

    def configure_optimizers(self):
        return {
            'optimizer': self.optimizer,
            'lr_scheduler': self.lr_scheduler,
            # 'monitor': 'training loss'
        }    

    def test_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        iou = self.IoU(out,y)
        f = F1(out,y)
        acc = accuracy(out,y)
        self.testIoUs.append(iou.detach().cpu().item())
        self.testF1s.append(f.detach().cpu().item())
        self.testaccs.append(acc.detach().cpu().item())

    def test_epoch_end(self, outputs):
        IoUs = np.array(self.testIoUs).mean()
        f = np.array(self.testF1s).mean()
        acc = np.array(self.testaccs).mean()
        logs = {'IoU': IoUs, 'results': (IoUs, f, acc)}
        return {'results':(IoUs, f), 'F1':f,'progress_bar': logs}

class UNet(pl.LightningModule):
    def __init__(self, lr=1e-4, n_channels=3, n_classes=1, return_features=False, mode='classic'):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.return_features = return_features
        self.mode = mode
        if mode == 'classic' or 'classic-backbone' in mode:
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
        elif mode == 'small':
            self.inc = DoubleConv(self.n_channels, 8)
            self.down1 = Down(8, 16)
            self.down2 = Down(16, 32)
            self.down3 = Down(32, 64)
            self.down4 = Down(64, 128)
            self.up1 = Up(128, 64)
            self.up2 = Up(64, 32)
            self.up3 = Up(32, 16)
            self.up4 = Up(16, 8)
            self.outc = OutConv(8, 1)
        elif mode == 'classic-dilation':
            self.inc = DoubleConv(self.n_channels, 64)
            self.down1 = Down(64, 128)
            self.down2 = Down(128, 256)
            self.down3 = Down(256, 512)

            self.dil1 = DilationBlock(512, 1024, 1, 1)
            self.dil2 = DilationBlock(512, 1024, 2, 2)
            self.dil3 = DilationBlock(512, 1024, 4, 4)
            self.dil4 = DilationBlock(512, 1024, 8, 8)
            self.dil5 = DilationBlock(512, 1024, 16, 16)

            self.up1 = Up(1024, 512)
            self.up2 = Up(512, 256)
            self.up3 = Up(256, 128)
            self.up4 = Up(128, 64)
            self.outc = OutConv(64, 1)

        elif mode == 'deeplab':
            # load deeplab
            self.model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=False, num_classes=n_classes)
            # change input channels of first convolution
            self.model.backbone.conv1 = nn.Conv2d(n_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),
                                                  bias=False)
            # register forward hook to be able to extract intermediate features
            self.features = None
            self.model.classifier[3].register_forward_hook(self.get_features())

        elif mode == 'segnet':
            self.model = SegNet(n_channels=n_channels, num_classes=1, pretrained=True, return_feats=return_features)

        elif mode == 'resnetduchdc':
            self.model = ResNetDUCHDC(n_channels=n_channels, num_classes=1, pretrained=True)

        elif mode == 'resnetduc':
            self.model = ResNetDUC(n_channels=n_channels, num_classes=1, pretrained=True)

    def get_features(self):
        def hook(model, input, output):
            self.features = output
        return hook

    def forward(self, x, backbone_feats=None):
        if 'classic-backbone' in self.mode and backbone_feats is not None:
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            x5 = x5 + backbone_feats
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
            logits = self.outc(x)
            if self.return_features:
                return x, logits
            else:
                return logits
        elif self.mode == 'classic' or self.mode == 'small' or ('classic-backbone' in self.mode and backbone_feats is None):
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
        elif self.mode == 'classic-dilation':
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            d1 = self.dil1(x4)
            d2 = self.dil2(x4)
            d3 = self.dil3(x4)
            d4 = self.dil4(x4)
            d5 = self.dil5(x4)
            x5 = d1 + d2 + d3 + d4 + d5
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
            logits = self.outc(x)
            if self.return_features:
                return x, logits
            else:
                return logits

        elif self.mode == 'deeplab':

            logits = self.model(x)['out']
            features = torch.nn.Upsample(mode='bilinear', size=x.shape[-1], align_corners=True)(self.features)

            if self.return_features:
                return features, logits
            else:
                return logits

        elif self.mode == 'segnet':
            if self.return_features:
                features, logits = self.model(x)
                return features, logits
            else:
                logits = self.model(x)
                return logits

        elif self.mode == 'resnetduchdc' or self.mode == 'resnetduc':
            logits = self.model(x)
            return logits
