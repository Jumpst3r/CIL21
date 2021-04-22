import pytorch_lightning as pl
import torch
import torch.nn as nn
from helpers import get_accuracy
import cv2
from .blocks import conv_block


class PatchNetBlock(nn.Module):

    def __init__(self):
        super(PatchNetBlock, self).__init__()

        self.conv_down_1 = conv_block(64, 64, 3)
        self.conv_down_2 = conv_block(64, 128, 3)
        self.conv_down_3 = conv_block(128, 256, 3)
        self.conv_down_4 = conv_block(256, 512, 3)
        self.low1 = conv_block(512, 512, 3)
        self.low2 = conv_block(512, 512, 3)

        self.max_pool = nn.MaxPool2d(kernel_size=2)

        self.conv_out = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, padding=0)

    def forward(self, x):

        x = self.conv_down_1(x)
        x = self.max_pool(x)

        x = self.conv_down_2(x)
        x = self.max_pool(x)

        x = self.conv_down_3(x)
        x = self.max_pool(x)

        x = self.conv_down_4(x)
        x = self.max_pool(x)

        x = self.low1(x)
        x = self.low2(x)

        out = self.conv_out(x)
        pred = torch.sigmoid(out)

        return pred


class PatchNet(pl.LightningModule):

    def __init__(self, lr=1e-4, stacks=1):
        super(PatchNet, self).__init__()

        if stacks != 1:
            raise RuntimeError("stacks /= 1 not yet supported")

        self.lr = lr
        self.stacks = stacks
        self.loss_fn = nn.BCELoss(reduction='sum')

        self.conv_pre = conv_block(3, 64, 3)
        self.patchnet = PatchNetBlock()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=80, gamma=0.5)

    def forward(self, x):
        I = self.conv_pre(x)
        pred = self.patchnet(I)
        return pred

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        loss = self.loss_fn(out, y)
        self.log('train_loss', loss)
        self.log('lr', self.optimizer.param_groups[0]['lr'])

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)

        # validation loss
        loss = self.loss_fn(pred, y)
        self.log('val_loss', loss)

        # validation accuracy (per pixel since 1 pixel = 1 16x16 patch)
        thresh = 0.5
        pred_thresh = pred.clone()
        pred_thresh[pred_thresh > thresh] = 1
        pred_thresh[pred_thresh < 1] = 0
        num_pixels = torch.prod(torch.tensor(y.shape))
        num_corr = torch.sum((pred_thresh == y).int())
        acc = num_corr / num_pixels
        self.log('val_acc', acc)

        # log images
        for i in range(pred.shape[0]):
            self.logger.experiment.log_image(x[i].detach().cpu().numpy().transpose(1,2,0), name='{}_{}_img'.format(batch_idx, i))
            self.logger.experiment.log_image(pred[i, 0].detach().cpu().numpy(), name='{}_{}_sigmoid'.format(batch_idx, i))
            self.logger.experiment.log_image(pred_thresh[i, 0].detach().cpu().numpy(), name='{}_{}_thresh'.format(batch_idx, i))
            self.logger.experiment.log_image(y[i, 0].detach().cpu().numpy(), name='{}_{}_gt'.format(batch_idx, i))

    def configure_optimizers(self):
        return {
            'optimizer': self.optimizer,
            'lr_scheduler': self.lr_scheduler,
            'monitor': 'val_loss'
        }
