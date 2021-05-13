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
import argparse
import time


pl.seed_everything(2)

from sklearn.model_selection import KFold
import gc

if __name__ == '__main__':
    dataset = ArealDataset(root_dir_images='training/images/', root_dir_gt='training/groundtruth/',
                           target_size=(128, 128))
    cross_val = 4
    kf = KFold(n_splits=cross_val)

    epochs = 200
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

    parser = argparse.ArgumentParser()
    parser.add_argument("key", help="model which should be trained", type=str)
    parser.add_argument("mode", help="enter \"eval\" or \"train\" to eval with CV or train on whole set for kaggle sub", type=str)
    key = vars(parser.parse_args())['key']
    mode = vars(parser.parse_args())['mode']

    def get_trainer():
        if torch.cuda.is_available():
            return pl.Trainer(max_epochs=epochs, deterministic=True, progress_bar_refresh_rate=0, logger=False)
        else:
            return pl.Trainer(max_epochs=epochs, gpus=1, deterministic=True, progress_bar_refresh_rate=0, logger=False) # precision=16, stochastic_weight_avg=True,


    #for key in seg_models.keys():
    def eval(key):
        print("training: ", key)
        fold = 0
        iou = np.zeros((cross_val, epochs))
        f1 = np.zeros((cross_val, epochs))
        for train_indices_plain, test_indices_plain in kf.split(dataset):
            train_indices = [idx[i] for i in train_indices_plain]
            test_indices = [idx[i] for i in test_indices_plain]
            train_dataset = torch.utils.data.dataset.Subset(dataset, train_indices)
            test_dataset = torch.utils.data.dataset.Subset(dataset, test_indices)
            test_dataset.applyTransforms = False
            train_dataloader = DataLoader(train_dataset, batch_size=5, pin_memory=True, num_workers=8)
            test_dataloader = DataLoader(test_dataset, batch_size=5, pin_memory=True, num_workers=8)
            model = VisionBaseline(seg_models[key], model_opts[key], loss[key], optimizer[key], optimizer_options[key], epochs)
            trainer = get_trainer()

            start = time.time()
            trainer.fit(model, train_dataloader, test_dataloader)
            end = time.time()
            print("fold: ", fold, " time: ", end-start, "s iou: ", model.val_iou, " f1: ", model.val_f1)

            #0th entry is sanity check, drop that
            iou[fold, :] = model.val_iou[1:-1]
            f1[fold, :] = model.val_f1[1:-1]

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

        np.save(key+"_iou", iou)
        np.save(key+"_f1", f1)

    def train(key):
        print("training for kaggle eval: ", key)
        train_dataset = torch.utils.data.dataset.Subset(dataset, [i for i in range(100)])
        train_dataloader = DataLoader(train_dataset, batch_size=5, pin_memory=True, num_workers=8)
        model = VisionBaseline(seg_models[key], model_opts[key], loss[key], optimizer[key], optimizer_options[key], epochs)
        trainer = get_trainer()

        start = time.time()
        trainer.fit(model, train_dataloader)
        end = time.time()
        print("time elapsed: ", end-start, "s epochs: ", epochs)
        #dct = model.state_dict()
        folder = '/cluster/scratch/fdokic/CIL21/'
        PATH = folder+ key+"_trained.pt"
        torch.save(model.state_dict(), PATH)

    if mode == "eval":
        eval(key)
    if mode == "train":
        train(key)



