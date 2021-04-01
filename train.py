import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from helpers import *
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.callbacks import ModelCheckpoint
from data.AerialShots import AerialShots
from models.RepeatedUNet import UNet
# from models.UNetLarge import UNet
import glob
import random

# hyperparam
train_split = 0.7
num_epochs = 1000
lr = 1e-4

# datasets
img_dir = 'training/training/images/'
labels_dir = 'training/training/groundtruth/'
img_paths = glob.glob(img_dir + '*.png')
label_paths = glob.glob(labels_dir + '*.png')
data_paths = list(zip(img_paths, label_paths))
random.shuffle(data_paths)
num_samples_total = len(img_paths)
num_train = int(train_split * num_samples_total)
data_paths_train = data_paths[:num_train]
data_paths_val = data_paths[num_train:]
train = AerialShots(data_paths_train, augment=True)
val = AerialShots(data_paths_val, augment=False)
train_dataloader = DataLoader(train, batch_size=6, shuffle=True, num_workers=8)
val_dataloader = DataLoader(val, batch_size=1, shuffle=False, num_workers=8)

# asd
#vis(next(iter(train_dataloader)))
#vis(next(iter(val_dataloader)))

checkpoint_callback = ModelCheckpoint(
        dirpath='./',
        filename='weights',
        monitor='val_loss',
        mode='min'
    )

unet = UNet(lr=lr)

logger = TensorBoardLogger("train_logs", name="UNet")

trainer = pl.Trainer(gpus=1,
                     checkpoint_callback=checkpoint_callback,
                     max_epochs=num_epochs,
                     logger=logger)

trainer.fit(unet, train_dataloader, val_dataloader)

# tensorboard --logdir train_logs/
