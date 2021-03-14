import os
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split, Dataset
from pytorch_lightning.metrics import functional as FM
import numpy as np
import glob
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data.sampler import WeightedRandomSampler
import matplotlib.pyplot as plt
from torchvision import transforms
import matplotlib.pyplot as plt
from models.unet import UNet

class RoadDataset(Dataset):

    def __init__(self, root_dir_images, root_dir_gt):
        self.images = glob.glob(root_dir_images + '*.png')
        self.gt = glob.glob(root_dir_gt + '*.png')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        impath = self.images[idx]
        gtpath = self.gt[idx]
        
        image =  transforms.ToTensor()(Image.open(impath))
        image -= image.mean()
        image /= image.std()
        gt = np.array(Image.open(gtpath))
        gt = np.where(np.array(gt) >=  1, 1., 0.)

        return image, torch.tensor(gt).unsqueeze(0)

def vizualize(x,y):
    """Visualize a tensor pair, where x is a tensor of shape [1,3,width,height] and y is a tensor of shape [1,width, height]
    """    
    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.axis('off')
    ax2.axis('off')
    ax1.imshow(np.moveaxis(np.array(x[0]),0,-1))
    ax2.imshow(y[0], cmap='binary_r')
    plt.show()


if __name__ == '__main__':
            
   
    dataset = RoadDataset(root_dir_images='training/training/images/',root_dir_gt='training/training/groundtruth/')
    num_samples_total = len(dataset)
    num_train = int(0.9 * num_samples_total)
    num_val = num_samples_total - num_train
    train, val = random_split(dataset, [num_train, num_val])

    train_dataloader = DataLoader(train, batch_size=5, num_workers=3)
    val_dataloader =  DataLoader(val, batch_size=5, num_workers=3)

    checkpoint_callback = ModelCheckpoint(
        dirpath='./',
        filename='weights',
        monitor='validation loss',
        mode='min'
    )
    fcn = UNet(3,1)
    trainer = pl.Trainer(checkpoint_callback=checkpoint_callback, max_epochs=1000, gpus=1)
    trainer.fit(fcn,train_dataloader,val_dataloader)
