from comet_ml import Experiment
import torch
import tensorboard
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from helpers import *
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.callbacks import ModelCheckpoint
from data.AerialShots import AerialShots
from models.StackedUNet import UNet
import glob
import random
from pytorch_lightning.loggers import CometLogger


def get_data_paths(img_dir, labels_dir):
    img_paths = sorted(glob.glob(img_dir + '*.png'))
    label_paths = sorted(glob.glob(labels_dir + '*.png'))
    data_paths = list(zip(img_paths, label_paths))
    # random.shuffle(data_paths) # I removed the shuffling before the split, so I can more accurately compare models
    return data_paths


# hyperparam
train_split = 0.7
num_epochs = 1000
lr = 1e-4
n_stacks = 4
batchsize = 8
img_size = 160
normalize = False
save_last = False
dropout = False
train_sets = 'cil'  # cil = CIL data only / cil_ms  = CIL data + massachusetts road dataset
weights_name = "weights_stacks-{}_bs-{}_imgsize-{}_trainsets-{}_lr-{}_dropout-{}_norm-{}".format(n_stacks, batchsize, img_size, train_sets, lr, dropout, normalize)

# additional stuff for he logger
data_filename = 'data/AerialShots.py'
model_filename = 'models/StackedUNet.py'
stacking_type = 'no weight sharing between stacks'
sched_type = 'StepLR(step_size=80, gamma=0.5)'

# data
data_paths_CIL_train = get_data_paths('training/training/images/', 'training/training/groundtruth/')
data_paths_CIL_val = get_data_paths('training/validation/images/', 'training/validation/groundtruth/')
data_paths_massachusetts = get_data_paths('../massachusetts/training/input/', '../massachusetts/training/output/')

# data sets
if train_sets == 'cil':
    train = AerialShots(data_paths_CIL_train, augment=True, normalize=normalize, img_size=img_size, label_size=img_size)
elif train_sets == 'ms':
    train = AerialShots(data_paths_massachusetts, augment=True, normalize=normalize, img_size=img_size, label_size=img_size)
elif train_sets == 'cil_ms':
    train = AerialShots(data_paths_CIL_train + data_paths_massachusetts, augment=True, normalize=normalize, img_size=img_size, label_size=img_size)
else:
    raise RuntimeError('Unknown training set code {}'.format(train_sets))
val = AerialShots(data_paths_CIL_val, augment=False, normalize=normalize, img_size=img_size, label_size=400)

# data loaders
train_dataloader = DataLoader(train, batch_size=batchsize, shuffle=True, num_workers=8)
val_dataloader = DataLoader(val, batch_size=1, shuffle=False, num_workers=8)

# asd
# iter_train = iter(train_dataloader)
# iter_val = iter(val_dataloader)
# vis(next(iter_train))
# vis(next(iter_val))

checkpoint_callback = ModelCheckpoint(
        dirpath='./',
        filename=weights_name,
        monitor='val_acc',
        mode='max',
        save_last=save_last
    )

# unet = UNet(lr=lr, stacks=n_stacks)
unet = UNet.load_from_checkpoint('weights_stacks-4_bs-10_imgsize-160_trainsets-cil_lr-0.0001_savelast-True_norm-False.ckpt')

comet_logger = CometLogger(
    api_key="tz8fhwoMrSi1d1s4tcKDME5uw",
    project_name="CIL2",
)
comet_logger.experiment.set_name(weights_name)
comet_logger.experiment.log_code(file_name=data_filename)
comet_logger.experiment.log_code(file_name=model_filename)

comet_logger.log_hyperparams(
    {
        "type of stacking": stacking_type,
        "number of UNET stacks": n_stacks,
        "initial lr": lr,
        "lr scheduler": sched_type,
        "training data": train_sets,
        "training set size": len(train),
        "name": weights_name,
        "batch size": batchsize,
        "training image size": img_size,
        "save_last": save_last,
        "normalize": normalize,
        "dropout": dropout
    }
)

trainer = pl.Trainer(gpus=1,
                     checkpoint_callback=checkpoint_callback,
                     max_epochs=num_epochs,
                     logger=comet_logger)

trainer.fit(unet, train_dataloader, val_dataloader)

# tensorboard --logdir train_logs/
