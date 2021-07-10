import os
import torch
from torch import nn
from torch.jit import export_opnames
import torch.nn.functional as F
from PIL import Image,ImageOps
import pytorch_lightning as pl
from torch.utils import data
from torch.utils.data import DataLoader, random_split, Dataset
from pytorch_lightning.metrics import functional as FM
import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from torchvision import transforms
import matplotlib.pyplot as plt
from models.unet import StackedUNet
# from models.unet.janikunet import StackedUNetPL
from tqdm import tqdm
from pytorch_lightning.loggers import CometLogger
import pytorch_lightning
from sklearn.model_selection import KFold
from dataset import ArealDataset, dataset_transform

from models.unet.utils import F1, accuracy

pl.seed_everything(2)

if __name__ == '__main__':

    dataset = ArealDataset(root_dir_images='training/images/',root_dir_gt='training/groundtruth/', target_size=(320,320))
    
    num_samples_total = len(dataset)
    num_train = int(0.7 * num_samples_total)
    num_val = num_samples_total - num_train
    train, val = random_split(dataset, [num_train, num_val])


    train_dataloader = DataLoader(train, batch_size=3, num_workers=8)
    val_dataloader =  DataLoader(val, batch_size=3, num_workers=8)
    
    # for x,y in train_dataloader: vizualize(x,y)

    checkpoint_callback = ModelCheckpoint(
        dirpath='./',
        filename='weights',
        monitor='IoU val',
        mode='max',
        save_top_k=5
    )
    # Default constructor parameters match the best performing configuration (6 stacks, hourglass style)
    model = StackedUNet()
    
    trainer = pl.Trainer(max_epochs=5000, gpus=1, stochastic_weight_avg=True, precision=16,
                         checkpoint_callback=checkpoint_callback, check_val_every_n_epoch=5, resume_from_checkpoint='weights-v4.ckpt')

    trainer.fit(model, train_dataloader, val_dataloader)
       
    

