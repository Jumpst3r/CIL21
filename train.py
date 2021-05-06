import comet_ml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CometLogger

from torch.utils.data import DataLoader

from models.stacked_unet_plwrapper import StackedUNetPL
from data.dataset import ArealDataset


# hyper parameters
batchsize = 1
acc_grad = 8
num_workers = 8
n_stacks = 4
num_epochs = 1000
lr = 1e-4
bilinear = True
image_size = 608
precision = 16
norm = 'batch'  # instance / batch
weights_name = 'StackedUNet_bs-{}_ns-{}_ac-{}_lr-{}_bil-{}_is-{}_p-{}_n-{}'\
    .format(batchsize, n_stacks, acc_grad, lr, bilinear, image_size, precision, norm)
# data loaders
trainds = ArealDataset('training/train/images/', 'training/train/groundtruth/', target_size=(image_size, image_size))
valds = ArealDataset('training/val/images/', 'training/val/groundtruth/', target_size=(image_size, image_size),
                     means=trainds.means, stds=trainds.stds, val=True)
train_dataloader = DataLoader(trainds, batch_size=batchsize, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
val_dataloader = DataLoader(valds, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)

# model saver
checkpoint_callback = ModelCheckpoint(
        dirpath='./',
        filename=weights_name,
        monitor='val_acc',
        mode='max',
    )

# logging
comet_logger = CometLogger(
    api_key="tz8fhwoMrSi1d1s4tcKDME5uw",
    project_name="StackedUNet",
)
comet_logger.experiment.set_name(weights_name)

# model
model = StackedUNetPL(lr=lr, n_stacks=n_stacks, bilinear=bilinear, norm=norm)

# training
trainer = pl.Trainer(gpus=1,
                     checkpoint_callback=checkpoint_callback,
                     max_epochs=num_epochs,
                     logger=comet_logger,
                     accumulate_grad_batches=acc_grad,
                     precision=precision
                     )
trainer.fit(model, train_dataloader, val_dataloader)