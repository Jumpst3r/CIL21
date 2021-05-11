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
from baseline_models import FcnResNet50

pl.seed_everything(2)

from sklearn.model_selection import KFold
import gc

if __name__ == '__main__':
    dataset = ArealDataset(root_dir_images='training/images/', root_dir_gt='training/groundtruth/',
                           target_size=(128, 128))
    kf = KFold(n_splits=4)
    val_IoU = []
    val_F1 = []

    np.random.seed(2) # just in case global seed doesnt cover numpy
    idx =  np.random.permutation(np.arange(100))
    for train_indices_plain, test_indices_plain in kf.split(dataset):
        train_indices = [idx[i] for i in train_indices_plain]
        test_indices = [idx[i] for i in test_indices_plain]
        train_dataset = torch.utils.data.dataset.Subset(dataset, train_indices)
        test_dataset = torch.utils.data.dataset.Subset(dataset, test_indices)
        test_dataset.applyTransforms = False
        train_dataloader = DataLoader(train_dataset, batch_size=5, pin_memory=True, num_workers=8)
        test_dataloader = DataLoader(test_dataset, batch_size=5, pin_memory=True, num_workers=8)
        model = FcnResNet50(1e-4) #StackedUNet(lr=1e-4, nb_blocks=1)
        trainer = pl.Trainer(max_epochs=3, deterministic=True) # gpus=1, precision=16, stochastic_weight_avg=True,
        trainer.fit(model, train_dataloader)
        results = trainer.test(model, test_dataloader, verbose=False)[0]
        #val_IoU.append(results['results'][0])
        #val_F1.append(results['results'][1])
        del model
        del trainer
        del train_dataset
        del test_dataset
        del train_dataloader
        del test_dataloader
        gc.collect()
        #torch.cuda.empty_cache()

    #print('mean/std IoU:', np.array(val_IoU).mean(), np.array(val_IoU).std())
    #print('mean/std F1:', np.array(val_F1).mean(), np.array(val_F1).std())



