import os
# from comet_ml import Experiment
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image, ImageOps
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split, Dataset
from pytorch_lightning.metrics import functional as FM
import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision import transforms
#import matplotlib.pyplot as plt
#from models.unet import StackedUNet
# from models.unet.janikunet import StackedUNetPL
#from tqdm import tqdm
from pytorch_lightning.loggers import CometLogger
import pytorch_lightning
from dataset import ArealDataset
from baseline_models import *
from torchvision.models.segmentation import fcn_resnet50, fcn_resnet101, deeplabv3_resnet50, deeplabv3_resnet101

pl.seed_everything(2)

from sklearn.model_selection import KFold
import gc

if __name__ == '__main__':
    dataset = ArealDataset(root_dir_images='training/images/', root_dir_gt='training/groundtruth/',
                           target_size=(128, 128))
    cross_val = 4
    kf = KFold(n_splits=cross_val)

    epochs = 1
    base_model_options = dict(pretrained=False, progress=True, num_classes=1)
    base_adam_options = dict(lr=1e-4, weight_decay=1e-5)
    seg_models = {"fcn_resnet50": fcn_resnet50, "fcn_resnet101": fcn_resnet101,
                  "deeplabv3_resnet50": deeplabv3_resnet50, "deeplabv3_resnet101": deeplabv3_resnet101}
    model_opts = {"fcn_resnet50": base_model_options, "fcn_resnet101": base_model_options,
                  "deeplabv3_resnet50": base_model_options, "deeplabv3_resnet101": base_model_options}
    loss = {"fcn_resnet50": F.binary_cross_entropy_with_logits, "fcn_resnet101": F.binary_cross_entropy_with_logits,
                  "deeplabv3_resnet50": F.binary_cross_entropy_with_logits, "deeplabv3_resnet101": F.binary_cross_entropy_with_logits}
    optimizer = {"fcn_resnet50": torch.optim.Adam, "fcn_resnet101": torch.optim.Adam,
                  "deeplabv3_resnet50": torch.optim.Adam, "deeplabv3_resnet101": torch.optim.Adam}
    optimizer_options = {"fcn_resnet50": base_adam_options, "fcn_resnet101": base_adam_options,
                  "deeplabv3_resnet50": base_adam_options, "deeplabv3_resnet101": base_adam_options}

    val_IoU = []
    val_F1 = []

    np.random.seed(2) # just in case global seed doesnt cover numpy
    idx =  np.random.permutation(np.arange(100))
    """
    (fold, epoch, batch/epoch) -> (fold, epoch) (only store median or mean)
    """
    for key in seg_models.keys():
        print("training: ", key)
        fold = 0
        for train_indices_plain, test_indices_plain in kf.split(dataset):
            train_indices = [idx[i] for i in train_indices_plain]
            test_indices = [idx[i] for i in test_indices_plain]
            train_dataset = torch.utils.data.dataset.Subset(dataset, train_indices)
            test_dataset = torch.utils.data.dataset.Subset(dataset, test_indices)
            test_dataset.applyTransforms = False
            train_dataloader = DataLoader(train_dataset, batch_size=5, pin_memory=True, num_workers=8)
            test_dataloader = DataLoader(test_dataset, batch_size=5, pin_memory=True, num_workers=8)
            model = VisionBaseline(seg_models[key], model_opts[key], loss[key], optimizer[key], optimizer_options[key])
            d = dict(model=key)
            iou = np.zeros((cross_val, epochs))
            f1 = np.zeros((cross_val, epochs))
            if torch.cuda.is_available():
                trainer = pl.Trainer(max_epochs=1, gpus=1, precision=16, stochastic_weight_avg=True, deterministic=True)
            else:
                trainer = pl.Trainer(max_epochs=1, deterministic=True)

            for i in range(epochs):
                trainer.fit(model, train_dataloader)
                results = trainer.test(model, test_dataloader, verbose=False)[0]
                iou[fold, i] = np.array(model.testIoU).mean()
                f1[fold, i] = np.array(model.testF1).mean()
                print(key + ", epoch: ", i, ", fold: ", fold, ", iou: ", iou[fold, i], ", f1: ", f1[fold, i] )

            fold += 1
            del model
            del trainer
            del train_dataset
            del test_dataset
            del train_dataloader
            del test_dataloader
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            d = {**d, key+"_iou":iou, key+"_f1":f1}
            np.save(key+"_iou", iou)
            np.save(key+"_f1", f1)

        #print('mean/std IoU:', np.array(val_IoU).mean(), np.array(val_IoU).std())
        #print('mean/std F1:', np.array(val_F1).mean(), np.array(val_F1).std())



