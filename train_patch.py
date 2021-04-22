from comet_ml import Experiment
import torch
import tensorboard
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from helpers import *
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.callbacks import ModelCheckpoint
from data.AerialShotsPatch import AerialShotsPatch
from data.DeepGlobePatch import DeepGlobePatch
# from models.PatchNet import PatchNet
from models.StackedPatchNet import PatchNet
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
num_epochs = 1000
lr = 1e-4
n_stacks = 5
batchsize = 4
img_size = 608
save_last = False
dropout = False
num_workers = 8
train_sets = 'cil'  # cil / deepglobe
stacking_type = 'stack'
weights_name = "patchnet_{}-{}_bs-{}_imgsize-{}_trainsets-{}_lr-{}_dropout-{}_ARCHTEST".format(stacking_type, n_stacks, batchsize, img_size, train_sets, lr, dropout)

# additional stuff for he logger
data_filename = 'data/AerialShotsPatch.py'
model_filename = 'models/StackedPatchNet.py'
sched_type = 'StepLR(step_size=80, gamma=0.5)'

# data
if train_sets == 'cil':
    data_paths_CIL_train = get_data_paths('training_patch/training/images/', 'training_patch/training/groundtruth/')
    data_paths_CIL_val = get_data_paths('training_patch/validation/images/', 'training_patch/validation/groundtruth/')
    train = AerialShotsPatch(data_paths_CIL_train, augment=True, img_size=img_size)
    val = AerialShotsPatch(data_paths_CIL_val, augment=False, img_size=img_size, val=True)
elif train_sets == 'deepglobe':
    img_paths_deepglobe = glob.glob(r'../deepglobe/train/*_sat.jpg')
    img_paths_deepglobe_train = img_paths_deepglobe[:int(len(img_paths_deepglobe)*0.9)]
    img_paths_deepglobe_val = img_paths_deepglobe[int(len(img_paths_deepglobe)*0.9):]
    train = DeepGlobePatch(img_paths_deepglobe_train, augment=True, img_size=img_size)
    val = DeepGlobePatch(img_paths_deepglobe_val, augment=False, img_size=img_size, val=True)
else:
    raise RuntimeError('Unknown training set code {}'.format(train_sets))
train_dataloader = DataLoader(train, batch_size=batchsize, shuffle=True, num_workers=num_workers)
val_dataloader = DataLoader(val, batch_size=1, shuffle=False, num_workers=num_workers)

# iter_train = iter(train_dataloader)
# iter_val = iter(val_dataloader)
# asd
# vis(next(iter_train))
# vis(next(iter_val))

checkpoint_callback = ModelCheckpoint(
        dirpath='./',
        filename=weights_name,
        monitor='val_acc',
        mode='max',
        save_last=save_last
    )

patchnet = PatchNet(lr=lr, stacks=n_stacks)
# patchnet = PatchNet.load_from_checkpoint('patchnet_stack-2_bs-2_imgsize-608_trainsets-deepglobe_lr-0.0001_dropout-False.ckpt')

comet_logger = CometLogger(
    api_key="tz8fhwoMrSi1d1s4tcKDME5uw",
    project_name="deepglobe",
)
comet_logger.experiment.set_name(weights_name)
comet_logger.experiment.log_code(file_name=data_filename)
comet_logger.experiment.log_code(file_name=model_filename)

comet_logger.log_hyperparams(
    {
        "stacking_type": stacking_type,
        "number of stacks": n_stacks,
        "initial lr": lr,
        "lr scheduler": sched_type,
        "training data": train_sets,
        "training set size": len(train),
        "name": weights_name,
        "batch size": batchsize,
        "training image size": img_size,
        "save_last": save_last,
        "dropout": dropout
    }
)

trainer = pl.Trainer(gpus=1,
                     checkpoint_callback=checkpoint_callback,
                     max_epochs=num_epochs,
                     logger=comet_logger)

trainer.fit(patchnet, train_dataloader, val_dataloader)

# tensorboard --logdir train_logs/
