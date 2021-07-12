
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
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
        IoUs = np.array(self.testIoU).mean()
        f = np.array(self.testF1).mean()
        acc = np.array(self.testAcc).mean()
        logs = {'IoU': IoUs, 'results': (IoUs, f, acc)}
        return {'results': (IoUs, f, acc), 'F1': f, 'progress_bar': logs}


class VisionBaselineSet(VisionBaseline):
    def __init__(self, model, model_opts, loss, optimizer, opt_opts, epochs,
                 train_idx, test_idx, train_res, augment):
        super().__init__(model, model_opts, loss, optimizer, opt_opts, epochs)
        self.train_idx = train_idx
        self.test_idx = test_idx
        self.train_res = train_res
        self.augment = augment
