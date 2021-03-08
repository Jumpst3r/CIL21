import os
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split, TensorDataset
from pytorch_lightning.metrics import functional as FM
import numpy as np
import glob
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data.sampler import WeightedRandomSampler
import matplotlib.pyplot as plt
from feature_extractor import PATCH_SIZE
from f1_score import f1_loss

class BaselineMLP(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.HU_COUNT = 200

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, kernel_size=(3,3), out_channels=5),
            nn.ReLU(),
            nn.Conv2d(in_channels=5, kernel_size=(3,3), out_channels=10),
            nn.ReLU(),
            nn.Conv2d(in_channels=10, kernel_size=(3,3), out_channels=16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, kernel_size=(3,3), out_channels=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            nn.ReLU()
        )
        self.constant = 6*6*32
        self.classifier = nn.Sequential(
            nn.Linear(self.constant, self.HU_COUNT),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(self.HU_COUNT, self.HU_COUNT),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(self.HU_COUNT, 2),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(-1, self.constant)
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        loss = F.cross_entropy(out, y, weight=torch.tensor([1,1.]))
        self.log('training loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        loss = F.cross_entropy(out, y, weight=torch.tensor([1,1.]))
        _f1_loss = f1_loss(y,out)
        self.log('F1 val', _f1_loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == '__main__':
            
    TRAINING_DATA = 'patches/'
    np_arrs_features = sorted(glob.glob(TRAINING_DATA + '*fts.npy'))
    np_arrs_labels = sorted(glob.glob(TRAINING_DATA + '*labels.npy'))

    X = np.stack([np.load(np_arr) for np_arr in np_arrs_features]).reshape((-1,3,PATCH_SIZE,PATCH_SIZE))
    Y = np.stack([np.load(np_arr) for np_arr in np_arrs_labels]).reshape((-1))
    
    X_pos = X[Y==1]
    X_neg = X[Y==0]

    np.random.shuffle(X_neg)
    X_neg = X_neg[:X_pos.shape[0]]

    assert X_pos.shape[0] == X_neg.shape[0]

    Y_neg = np.zeros(X_neg.shape[0])
    Y_pos = np.ones(X_neg.shape[0])

    Y = np.concatenate((Y_neg,Y_pos))
    X = np.concatenate((X_neg, X_pos))
    
    print(X.mean())
    print(X.std()) 

    X -= X.mean()
    X /= X.std() 

    

    tensor_x = torch.Tensor(X)
    tensor_y = torch.Tensor(Y).to(torch.long)
    # print(torch.sum(tensor_y==0))
    # print(torch.sum(tensor_y==1))
    # print(tensor_x.shape, tensor_y.shape)

    dataset = TensorDataset(tensor_x,tensor_y)

    train, val = random_split(dataset, [int(0.8*X.shape[0]), X.shape[0]-int(0.8*X.shape[0])])

    checkpoint_callback = ModelCheckpoint(
        dirpath='./',
        filename='weights',
        monitor='F1 val',
        mode='max'
    )


    baseline = BaselineMLP()
    trainer = pl.Trainer(checkpoint_callback=checkpoint_callback, max_epochs=15)
    trainer.fit(baseline, DataLoader(train, batch_size=500), DataLoader(val, batch_size=500))