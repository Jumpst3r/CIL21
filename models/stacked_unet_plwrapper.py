import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

from .stacked_unet_model import StackedUNet
from .helpers import fig2img


class StackedUNetPL(pl.LightningModule):
    def __init__(self, lr=1e-4, n_stacks=4, bilinear=False, norm='instance'):
        super(StackedUNetPL, self).__init__()

        # the model
        self.stacked_unet = StackedUNet(n_channels=3, n_classes=1, n_stacks=n_stacks, bilinear=bilinear, norm=norm)

        # optimization parameters
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.loss_weights = [1] * n_stacks

        # denormalize for vis
        self.inv_normalize = transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.255]
        )

    def forward(self, x):
        logits_list = self.stacked_unet(x)
        return logits_list

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits_list = self.forward(x)
        loss = torch.tensor(0., device='cuda')
        for i, logits in enumerate(logits_list):
            loss += self.loss_weights[i] * self.loss_fn(logits, y)
        self.log('train_loss', loss)
        self.log('lr', self.optimizer.param_groups[0]['lr'])
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits_list = self.forward(x)
        logits = logits_list[-1]

        # validation loss just on output
        loss = self.loss_fn(logits, y)
        self.log('val_loss', loss)

        # per-pixel accuracy
        prob = torch.sigmoid(logits)
        thresh = prob.clone()
        thresh[prob > 0.5] = 1
        thresh[thresh < 1] = 0
        acc = torch.sum(thresh == y) / torch.prod(torch.tensor(y.shape))
        self.log('val_acc', acc)

        # plot
        for i in range(y.shape[0]):
            image = x[i].to(dtype=torch.float32)
            image = self.inv_normalize(image)
            image = image.detach().cpu().numpy().transpose(1, 2, 0)
            prediction = prob[i, 0].to(dtype=torch.float32).detach().cpu().numpy()
            gt = y[i, 0].detach().cpu().numpy()

            fig = plt.figure(figsize=(12, 3), frameon=False)
            ax = fig.add_subplot(1, 4, 1)
            ax.imshow(image)
            ax = fig.add_subplot(1, 4, 2)
            ax.imshow(prediction)
            ax = fig.add_subplot(1, 4, 3)
            prediction[prediction < 0.1] = np.nan
            ax.imshow(image)
            ax.imshow(prediction, alpha=0.7, interpolation='none', vmin=0)
            ax = fig.add_subplot(1, 4, 4)
            ax.imshow(image)
            gt[gt < 0.1] = np.nan
            ax.imshow(gt, alpha=0.7, interpolation='none', vmin=0)
            plt.close()
            im = fig2img(fig, f=1)
            self.logger.experiment.log_image(np.array(im), name=str(batch_idx))

    def configure_optimizers(self):
        return {
            'optimizer': self.optimizer,
            'monitor': 'val_acc'
        }
