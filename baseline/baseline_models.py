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
"""

from torchvision import models, transforms, datasets, utils
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from kaggle_dataset import KaggleSet
from sklearn.model_selection import KFold
import numpy as np

seg_models = ["fcn_resnet50", "fcn_resnet101", "deeplabv3_resnet50", "deeplabv3_resnet101", "deeplabv3_mobilenet_v3_large",
          "lraspp_mobilenet_v3_large"]

import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl
from utils import IoU, DiceBCELoss, IoULoss, FocalLoss, F1

class FcnResNet50(pl.LightningModule):
    def __init__(self, lr):
        super().__init__()
        self.model = models.segmentation.fcn_resnet50(pretrained=False, progress=True, num_classes=1).train()
        self.loss = F.binary_cross_entropy_with_logits
        self.IoU = IoU
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)
        self.testIoU = []
        self.testF1 = []

    def forward(self, x):
        out  = self.model(x)
        return out['out']

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)
        loss = self.loss(pred, y)
        self.log('training loss', loss)
        return loss

    """def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)
        loss = self.loss(pred, y)
        iou = self.IoU(pred, y)
        self.log('IoU val', iou)
        self.log('loss val', loss)
        return {'loss val': loss, 'IoU val': iou} """

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
        print("len:", len(self.testIoU), "IoU", IoU, "f", f)
        out = {'results': (IoU, f), 'F1': f, 'progress_bar': logs}
        return out

"""
        IoU = np.array(model.testIoU).mean()
        f = np.array(model.testF1).mean()
        logs = {'IoU': IoU, 'results': (IoU, f)}
        out = {'results': (IoU, f), 'F1': f, 'progress_bar': logs}
"""