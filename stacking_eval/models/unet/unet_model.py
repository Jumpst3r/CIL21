""" Full assembly of the parts to form the complete network """
# base underlying unet model adapted from https://github.com/milesial/Pytorch-UNet

import numpy as np
import torchvision

from .unet_parts import *
from .utils import F1, IoU, accuracy


class StackedUNet(pl.LightningModule):
    def __init__(self,
                 lr=1e-4,
                 nb_blocks=4,
                 unet_mode='classic-backbone',
                 stacking_mode='hourglass',
                 loss_mode='sum',
                 use_scheduler=True):
        """
        lr: learning rate
        nb_blocks: # iterative unet blocks
        unet_mode: architecture
            classic: use the classic UNet in each stack
            classic-backbone: first stack uses ResNet backbone
        stacking_mode: stacking approach
            hourglass: stacking similar to Stacked Hourglass paper
            simple: each block gets only the image & previous prediction
        loss_mode: how the combine losses from different stacks
            avg: mean between the last & the mean of the other blocks
            last: only apply loss at the last block
        """

        super(StackedUNet, self).__init__()

        # stacked UNet model
        if stacking_mode == 'hourglass':
            enc_dim = 64
            self.initBlock = UNet(n_channels=3,
                                  return_features=True,
                                  mode=unet_mode)
            self.blocks = nn.ModuleList(
                UNet(n_channels=enc_dim, return_features=True, mode=unet_mode)
                for _ in range(nb_blocks - 1))
            self.conv_up_input = nn.Conv2d(in_channels=3,
                                           out_channels=enc_dim,
                                           kernel_size=1,
                                           padding=0)
            self.conv_up_logits_init = nn.Conv2d(in_channels=1,
                                                 out_channels=enc_dim,
                                                 kernel_size=1,
                                                 padding=0)
            self.conv_up_logits = nn.ModuleList(
                nn.Conv2d(in_channels=1,
                          out_channels=enc_dim,
                          kernel_size=1,
                          padding=0) for _ in range(nb_blocks - 1))
        elif stacking_mode == 'simple':
            self.initBlock = UNet(n_channels=3,
                                  return_features=False,
                                  mode=unet_mode)
            self.blocks = nn.ModuleList(
                UNet(n_channels=4, return_features=False, mode=unet_mode)
                for _ in range(nb_blocks - 1))
        else:
            raise RuntimeError(
                'Unsupported stacking mode {} in StackedUNet()'.format(
                    stacking_mode))

        if 'backbone' in unet_mode:
            rs50 = torchvision.models.resnet50(pretrained=True)
            self.backbone = torch.nn.Sequential(rs50.conv1, rs50.bn1,
                                                rs50.relu, rs50.maxpool,
                                                rs50.layer1, rs50.layer2,
                                                rs50.layer3)

        # optimizer
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=self.lr,
                                          weight_decay=1e-5)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                            gamma=0.5,
                                                            step_size=100,
                                                            verbose=True)
        self.loss = F.binary_cross_entropy_with_logits

        # hyper parameters
        self.use_scheduler = use_scheduler
        self.training = True
        self.unet_mode = unet_mode
        self.stacking_mode = stacking_mode
        self.loss_mode = loss_mode
        self.nb_blocks = nb_blocks
        self.alpha = 0.5

        # for validation
        self.IoU = IoU
        self.testIoUs = []
        self.testF1s = []
        self.testaccs = []

    def forward(self, x):
        if self.stacking_mode == 'simple':
            # initial prediction from the image
            if 'backbone' in self.unet_mode:
                backbone_feats = self.backbone(x)
                y1 = self.initBlock(x, backbone_feats)
            else:
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
            for conv_up_logits_init, block in zip(self.conv_up_logits,
                                                  self.blocks):
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
        if self.nb_blocks == 1:
            loss = self.loss(yL[-1], y)
        else:
            if self.loss_mode == 'avg':
                loss = (1 - self.alpha) * self.loss(yL[-1], y) \
                       + self.alpha * sum([self.loss(yhat, y) for yhat in yL[:-1]]) / (self.nb_blocks - 1)
            elif self.loss_mode == 'sum':
                loss = sum([self.loss(yhat, y) for yhat in yL])
            elif self.loss_mode == 'last':
                loss = self.loss(yL[-1], y)
            else:
                raise RuntimeError(
                    'Unsupported loss mode {} in training_step()'.format(
                        self.los_mode))
        self.log('training loss', loss)
        self.log('lr', self.lr)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        loss = self.loss(out, y)
        iou = self.IoU(out, y)
        self.log('IoU val', iou)
        self.log('loss val', loss)
        return {'loss val': loss, 'IoU val': iou}

    def configure_optimizers(self):
        if self.use_scheduler:
            return {
                'optimizer': self.optimizer,
                'lr_scheduler': self.lr_scheduler,
            }
        else:
            return {
                'optimizer': self.optimizer,
            }

    def test_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        iou = self.IoU(out, y)
        f = F1(out, y)
        acc = accuracy(out, y)
        self.testIoUs.append(iou.detach().cpu().item())
        self.testF1s.append(f.detach().cpu().item())
        self.testaccs.append(acc.detach().cpu().item())

    def test_epoch_end(self, outputs):
        IoUs = np.array(self.testIoUs).mean()
        f = np.array(self.testF1s).mean()
        acc = np.array(self.testaccs).mean()
        logs = {'IoU': IoUs, 'results': (IoUs, f, acc)}
        return {'results': (IoUs, f), 'F1': f, 'progress_bar': logs}


class UNet(pl.LightningModule):
    def __init__(self,
                 n_channels=3,
                 n_classes=1,
                 return_features=False,
                 mode='classic'):
        """
        n_channels: # channels of the input
        n_classes: # channels of the output
        return_features: return the final feature maps
        mode: architecture mode (see StackedUNet)
        """
        super(UNet, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.return_features = return_features
        self.mode = mode

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

    def forward(self, x, backbone_feats=None):
        if self.mode == 'classic-backbone' and backbone_feats is not None:
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
        elif self.mode == 'classic' or (self.mode == 'classic-backbone'
                                        and backbone_feats is None):
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
        else:
            raise RuntimeError('Unsupported mode {} in UNet.forward()'.format(
                self.mode))
