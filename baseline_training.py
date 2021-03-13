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
from f1_score import IoU
from torchvision import transforms
import matplotlib.pyplot as plt

class BaselineMLP(pl.LightningModule):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(in_channels=3, kernel_size=(3,3), padding=(1,1), out_channels=16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, kernel_size=(3,3), padding=(1,1), out_channels=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            nn.Conv2d(in_channels=32, kernel_size=(3,3), padding=(1,1) ,out_channels=64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, kernel_size=(3,3),padding=(1,1) ,out_channels=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)), # 100x100
            nn.ConvTranspose2d(in_channels=128, kernel_size=(2,2), stride=(2,2), out_channels=64), # 200x200
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, kernel_size=(2,2), stride=(2,2), out_channels=128), # 400x400
            nn.ReLU(),
            nn.Conv2d(in_channels=128, kernel_size=(1,1), out_channels=1),
        )
       

    def forward(self, x):
        x = self.network(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        loss = F.binary_cross_entropy_with_logits(out, y)
        self.log('training loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        loss = F.binary_cross_entropy_with_logits(out, y)
        self.log('validation loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


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
    num_train = int(0.8 * num_samples_total)
    num_val = num_samples_total - num_train
    train, val = random_split(dataset, [num_train, num_val])

    train_dataloader = DataLoader(train, batch_size=3, num_workers=1)
    val_dataloader =  DataLoader(val, batch_size=3, num_workers=1)

    checkpoint_callback = ModelCheckpoint(
        dirpath='./',
        filename='weights',
        monitor='validation loss',
        mode='min'
    )

    baseline = BaselineMLP()
    trainer = pl.Trainer(checkpoint_callback=checkpoint_callback, max_epochs=200, gpus=1)
    trainer.fit(baseline,train_dataloader,val_dataloader)