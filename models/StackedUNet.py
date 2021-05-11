import pytorch_lightning as pl
import torch
import torch.nn as nn
from helpers import get_accuracy
import cv2
# from vggloss import VGGLoss


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

    # return nn.Sequential(
    #     nn.Conv2d(in_channels=in_c, out_channels=mid_c, kernel_size=k, padding=p),
    #     nn.BatchNorm2d(mid_c),
    #     nn.ReLU(),
    #     nn.Dropout2d(p=0.1),
    #     nn.Conv2d(in_channels=mid_c, out_channels=out_c, kernel_size=k, padding=p),
    #     nn.BatchNorm2d(out_c),
    #     nn.ReLU(),
    #     nn.Dropout2d(p=0.1)
    # )


def upconv_block(in_c, out_c, k, s):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels=in_c, out_channels=out_c, kernel_size=k, stride=s),
        nn.BatchNorm2d(out_c)
    )


class UNetBlock(nn.Module):
    def __init__(self):
        super(UNetBlock, self).__init__()

        # to bring prediction to 64 channels
        self.conv_post = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=1, padding=0)

        # downward path conv block
        self.conv_down_1 = conv_block(64, 64, 3)
        self.conv_down_2 = conv_block(64, 128, 3)
        self.conv_down_3 = conv_block(128, 256, 3)
        self.conv_down_4 = conv_block(256, 512, 3)

        # lowest convolution
        self.conv_bottom = conv_block(512, 1024, 3)

        # upward path conv blocks
        self.conv_up_1 = conv_block(1024, 512, 3)
        self.conv_up_2 = conv_block(512, 256, 3)
        self.conv_up_3 = conv_block(256, 128, 3)
        self.conv_up_4 = conv_block(128, 64, 3)

        # output
        self.conv_out = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, padding=0)

        # down/up sampling
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        # self.up = nn.Upsample(scale_factor=2, mode='nearest')
        # self.up_1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        # self.up_2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        # self.up_3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        # self.up_4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.up_1 = upconv_block(in_c=1024, out_c=512, k=2, s=2)
        self.up_2 = upconv_block(in_c=512, out_c=256, k=2, s=2)
        self.up_3 = upconv_block(in_c=256, out_c=128, k=2, s=2)
        self.up_4 = upconv_block(in_c=128, out_c=64, k=2, s=2)

    def forward(self, x):

        # downward pass
        c1 = self.conv_down_1(x)
        x = self.max_pool(c1)
        c2 = self.conv_down_2(x)
        x = self.max_pool(c2)
        c3 = self.conv_down_3(x)
        x = self.max_pool(c3)
        c4 = self.conv_down_4(x)
        x = self.max_pool(c4)

        # bottom
        x = self.conv_bottom(x)

        # up pass
        x = self.up_1(x)
        x = torch.cat([c4, x], dim=1)
        x = self.conv_up_1(x)
        x = self.up_2(x)
        x = torch.cat([c3, x], dim=1)
        x = self.conv_up_2(x)
        x = self.up_3(x)
        x = torch.cat([c2, x], dim=1)
        x = self.conv_up_3(x)
        x = self.up_4(x)
        x = torch.cat([c1, x], dim=1)
        P = self.conv_up_4(x)

        # output layer
        out = self.conv_out(P)
        pred = torch.sigmoid(out)

        # bring output prediction back to 64 channels
        F = self.conv_post(pred)

        return P, F, pred


class UNet(pl.LightningModule):

    def __init__(self, lr=1e-4, stacks=8):
        super(UNet, self).__init__()

        # optimization parameters
        self.lr = lr
        self.stacks = stacks
        self.loss_fn = nn.BCELoss(reduction='sum')
        # self.loss_fn = VGGLoss()
        self.loss_weights = [1] * self.stacks
        self.loss_weights[-1] = self.stacks - 1

        # to bring input to 64 channels
        self.conv_pre = conv_block(3, 64, 3)

        # UNet blocks
        self.unets = nn.ModuleList([
            nn.Sequential(
                UNetBlock(),
            ) for i in range(stacks)])

    def forward(self, x):

        predictions = []
        I = [None] * self.stacks
        P = [None] * self.stacks
        F = [None] * self.stacks
        pred = [None] * self.stacks

        # bring input image to 64 channels
        I[0] = self.conv_pre(x)
        # first unet block
        P[0], F[0], pred[0] = self.unets[0](I[0])

        for i in range(1, self.stacks):
            # combine features
            I[i] = I[i-1] + P[i-1] + F[i-1]
            # second unet block
            P[i], F[i], pred[i] = self.unets[i](I[i])

        return pred

    def training_step(self, batch, batch_idx):
        x, y = batch
        predictions = self.forward(x)
        loss = torch.tensor(0., device='cuda')
        for i, out in enumerate(predictions):
            loss += self.loss_weights[i] * self.loss_fn(out, y)
        self.log('train_loss', loss)
        self.log('lr', self.optimizer.param_groups[0]['lr'])
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        predictions = self.forward(x)
        out = predictions[-1]

        # compute validation loss
        if out.shape == y.shape:
            loss = self.loss_fn(out, y)
            self.log('val_loss', loss)
            self.logger.experiment.log_image(out[0][0].detach().cpu().numpy())

        # compute validation accuracy
        thresh = 0.8
        acc = 0
        for i in range(out.shape[0]):
            pred = out[i, 0].detach().cpu().numpy()
            gt = y[i, 0].detach().cpu().numpy()
            pred = cv2.resize(pred, gt.shape, interpolation=cv2.INTER_NEAREST)
            self.logger.experiment.log_image(pred, name='{}_{}_sigmoid'.format(batch_idx, i))
            pred[pred > thresh] = 1
            pred[pred < 1] = 0
            self.logger.experiment.log_image(pred, name='{}_{}_thresh'.format(batch_idx, i))
            acc += get_accuracy(pred, gt)
        acc /= out.shape[0]
        self.log('val_acc', acc)

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=80, gamma=0.5)
        # self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, verbose=True)
        # self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=0.1, epochs=1000, steps_per_epoch=70, verbose=False)
        return {
            'optimizer': self.optimizer,
            'lr_scheduler': self.lr_scheduler,
            'monitor': 'val_loss'
        }