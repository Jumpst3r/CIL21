import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from helpers import *
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.callbacks import ModelCheckpoint
from data.AerialShots import AerialShots
from models.UNet import UNet

# hyperparam
train_split = 0.7
num_epochs = 100
lr = 1e-4

dataset = AerialShots(img_dir='training/training/images/',
                      labels_dir='training/training/groundtruth/')

num_samples_total = len(dataset)
num_train = int(train_split * num_samples_total)
num_val = num_samples_total - num_train
train, val = random_split(dataset, [num_train, num_val])

train_dataloader = DataLoader(train, batch_size=2, shuffle=True, num_workers=0)
val_dataloader = DataLoader(val, batch_size=2, shuffle=False, num_workers=0)

# vis(next(iter(train_dataloader)))

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