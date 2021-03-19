import pytorch_lightning as pl
import torch
import torch.nn as nn


def conv_block(in_c, out_c, k=3, p=1):
    mid_c = out_c
    return nn.Sequential(
        nn.Conv2d(in_channels=in_c, out_channels=mid_c, kernel_size=k, padding=p),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=mid_c, out_channels=out_c, kernel_size=k, padding=p),
        nn.ReLU(inplace=True)
    )


class UNet(pl.LightningModule):

    def __init__(self, lr=1e-3):
        super(UNet, self).__init__()

        # optimization parameters
        self.lr = lr
        self.loss_fn = nn.BCELoss(reduction='sum')

        # downward path conv block
        self.conv_down_1 = conv_block(3, 64, 3)
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
        self.up_1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.up_2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.up_3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.up_4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)

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
        x = self.conv_up_4(x)

        # output layer
        x = self.conv_out(x)

        return torch.sigmoid(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        loss = self.loss_fn(out, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        loss = self.loss_fn(out, y)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        # TODO: set params
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, verbose=True)
        return {
            'optimizer': self.optimizer,
            'lr_scheduler': self.lr_scheduler,
            'monitor': 'val_loss'
        }