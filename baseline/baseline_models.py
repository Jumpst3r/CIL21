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
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms, utils
from torchvision.models.segmentation import (deeplabv3_resnet50,
                                             deeplabv3_resnet101, fcn_resnet50,
                                             fcn_resnet101)

from utils import F1, IoU, accuracy

base_model_options = dict(pretrained=False, progress=True, num_classes=1)
base_adam_options = dict(lr=1e-4, weight_decay=1e-5)
seg_models = {
    "fcn_resnet50": fcn_resnet50,
    "fcn_resnet101": fcn_resnet101,
    "deeplabv3_resnet50": deeplabv3_resnet50,
    "deeplabv3_resnet101": deeplabv3_resnet101
}
model_opts = {
    "fcn_resnet50": base_model_options,
    "fcn_resnet101": base_model_options,
    "deeplabv3_resnet50": base_model_options,
    "deeplabv3_resnet101": base_model_options
}
loss = {
    "fcn_resnet50": F.binary_cross_entropy_with_logits,
    "fcn_resnet101": F.binary_cross_entropy_with_logits,
    "deeplabv3_resnet50": F.binary_cross_entropy_with_logits,
    "deeplabv3_resnet101": F.binary_cross_entropy_with_logits
}
optimizer = {
    "fcn_resnet50": torch.optim.Adam,
    "fcn_resnet101": torch.optim.Adam,
    "deeplabv3_resnet50": torch.optim.Adam,
    "deeplabv3_resnet101": torch.optim.Adam
}
optimizer_options = {
    "fcn_resnet50": base_adam_options,
    "fcn_resnet101": base_adam_options,
    "deeplabv3_resnet50": base_adam_options,
    "deeplabv3_resnet101": base_adam_options
}


class VisionBaseline(pl.LightningModule):
    def __init__(self, model, model_opts, loss, optimizer, opt_opts, epochs):
        super().__init__()
        self.model = model(**model_opts).train()
        self.loss = loss
        opts = {"params": self.parameters(), **opt_opts}
        self.optimizer = optimizer(**opts)
        self.testIoU = []
        self.testF1 = []
        self.testAcc = []
        self.IoU = IoU
        self.acc = accuracy
        self.curr_epoch = 0
        self.curr_fold = 0
        self.val_iou = np.zeros((epochs + 2))
        self.val_f1 = np.zeros((epochs + 2))
        self.val_acc = np.zeros((epochs + 2))

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
        return {'optimizer': self.optimizer, 'monitor': 'training loss'}

    def test_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        iou = self.IoU(out, y)
        f = F1(out, y)
        acc = self.acc(out, y)
        self.testIoU.append(iou.detach().cpu().item())
        self.testF1.append(f.detach().cpu().item())
        self.testAcc.append(acc.detach().cpu().item())

    def test_epoch_end(self, outputs):
        IoU = np.array(self.testIoU).mean()
        f = np.array(self.testF1).mean()
        acc = np.array(self.testAcc).mean()
        logs = {'IoU': IoU, 'results': (IoU, f, acc)}
        self.testF1 = []
        self.testIoU = []
        self.testAcc = []
        self.val_f1[self.curr_epoch] = f
        self.val_iou[self.curr_epoch] = IoU
        self.val_acc[self.curr_epoch] = acc

        self.curr_epoch += 1
        out = {'results': (IoU, f), 'F1': f, 'progress_bar': logs}
        return out

    def validation_step(self, batch, batch_idx):
        self.test_step(batch, batch_idx)

    def validation_epoch_end(self, outputs):
        self.test_epoch_end(outputs)


class VisionBaselineSet(VisionBaseline):
    def __init__(self, model, model_opts, loss, optimizer, opt_opts, epochs,
                 train_idx, test_idx, train_res, augment):
        super().__init__(model, model_opts, loss, optimizer, opt_opts, epochs)
        self.train_idx = train_idx
        self.test_idx = test_idx
        self.train_res = train_res
        self.augment = augment
