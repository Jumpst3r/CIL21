"""
Torchvision Models semantinc segmentation:
    - FCN ResNet50
    - FCN ResNet101
    - DeepLabV3 ResNet50
    - DeepLabV3 ResNet101
    - DeepLabV3 MobileNetV3
    - Lite R-ASPP MobileNetV3

Dataset:
    - Kaggle training set
    - Kaggle test set

performance measure function
    - IoU

Evaluation method:
    - 4 fold Cross-validation, mean of performance
    - full training set, Kaggle submission performance



import torch
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from kaggle_dataset import KaggleSet
from sklearn.model_selection import KFold
"""

import numpy as np
from torchvision import models, transforms, datasets, utils
from torchvision.models.segmentation import fcn_resnet50, fcn_resnet101, deeplabv3_resnet50, deeplabv3_resnet101
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from utils import IoU, DiceBCELoss, IoULoss, FocalLoss, F1

class VisionBaseline(pl.LightningModule):
    def __init__(self, model, model_opts, loss, optimizer, opt_opts, epochs):
        super().__init__()
        self.model = model(**model_opts).train()
        self.loss = loss
        opts = {"params": self.parameters(), **opt_opts}
        self.optimizer = optimizer(**opts)
        self.testIoU = []
        self.testF1 = []
        self.IoU = IoU
        self.curr_epoch = 0
        self.curr_fold = 0
        self.val_iou = np.zeros((epochs+2))
        self.val_f1 = np.zeros((epochs+2))

    def forward(self, x):
        out = self.model(x)
        return out['out']

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)
        loss = self.loss(pred, y)
        self.log('training loss', loss)
        return loss

    def configure_optimizers(self):
        return {
            'optimizer': self.optimizer,
            'monitor': 'training loss'
        }

    def test_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        iou = self.IoU(out, y)
        f = F1(out, y)
        self.testIoU.append(iou.detach().cpu().item())
        self.testF1.append(f.detach().cpu().item())

    def test_epoch_end(self, outputs):
        IoU = np.array(self.testIoU).mean()
        f = np.array(self.testF1).mean()
        logs = {'IoU': IoU, 'results': (IoU, f)}
        print("len:", len(self.testIoU), "curr_epoch: ", self.curr_epoch,"IoU", IoU, "f", f)
        self.testF1 = []
        self.testIoU = []
        self.val_f1[self.curr_epoch] = f
        self.val_iou[self.curr_epoch] = IoU
        self.curr_epoch += 1
        out = {'results': (IoU, f), 'F1': f, 'progress_bar': logs}
        return out

    def validation_step(self, batch, batch_idx):
        self.test_step(batch, batch_idx)

    def validation_epoch_end(self, outputs):
        self.test_epoch_end(outputs)
