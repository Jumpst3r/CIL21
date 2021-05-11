import pytorch_lightning as pl
import torch
import torch.nn as nn
from helpers import get_accuracy
import cv2
from .blocks import conv_block
import torchvision.models as models


class PatchNetBlock(nn.Module):

    def __init__(self):
        super(PatchNetBlock, self).__init__()

        self.conv_block_1 = conv_block(1024, 512, 3)
        self.conv_block_2 = conv_block(512, 256, 3)
        self.conv_block_3 = conv_block(256, 512, 3)
        self.conv_block_4 = conv_block(512, 1024, 3)

        self.conv_out = nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=1, padding=0)

        self.conv_post_pred = nn.Conv2d(in_channels=1, out_channels=1024, kernel_size=1, padding=0)

    def forward(self, x):

        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.conv_block_4(x)

        out = self.conv_out(x)
        pred = torch.sigmoid(out)

        P = x
        F = self.conv_post_pred(pred)

        return P, F, pred


class PatchNet(pl.LightningModule):

    def __init__(self, lr=1e-4, stacks=5):
        super(PatchNet, self).__init__()

        self.lr = lr
        self.stacks = stacks
        self.loss_fn = nn.BCELoss(reduction='sum')

        # backbone
        self.backbone = nn.Sequential(*list(models.resnet101(pretrained=True).cuda().children())[:-3])
        self.backbone.train()

        # repeated nets
        self.patchnets = nn.ModuleList([
            nn.Sequential(
                PatchNetBlock(),
            ) for i in range(stacks)])

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=80, gamma=0.5)

    def forward(self, x):
        I = [None] * self.stacks
        P = [None] * self.stacks
        F = [None] * self.stacks
        pred = [None] * self.stacks

        # pass through backbone to encode into feature space
        I[0] = self.backbone(x)

        # first block
        P[0], F[0], pred[0] = self.patchnets[0](I[0])

        for i in range(1, self.stacks):
            # combine features
            I[i] = I[i-1] + P[i-1] + F[i-1]
            # second patchnet block
            P[i], F[i], pred[i] = self.patchnets[i](I[i])

        return pred

    def training_step(self, batch, batch_idx):
        x, y = batch
        predictions = self.forward(x)

        loss = torch.tensor(0., device='cuda')
        for i, out in enumerate(predictions):
            loss += self.loss_fn(out, y)

        self.log('train_loss', loss)
        self.log('lr', self.optimizer.param_groups[0]['lr'])

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        predictions = self.forward(x)
        pred = predictions[-1]

        # # validation loss
        # loss = self.loss_fn(pred, y)
        # self.log('val_loss', loss)

        # validation accuracy (per pixel since 1 pixel = 1 16x16 patch)
        thresh = 0.5
        pred_thresh = pred.clone()
        pred_thresh[pred_thresh > thresh] = 1
        pred_thresh[pred_thresh < 1] = 0
        pred_thresh = torch.nn.functional.interpolate(pred_thresh, mode='nearest', size=(38, 38))
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
