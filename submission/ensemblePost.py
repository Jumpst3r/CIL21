import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from albumentations.augmentations.functional import rot90
from albumentations.pytorch.transforms import ToTensorV2
from torchvision import transforms

'''
Two goals:

 - Implement voting style ensemble techniques
 - Implement test time augmentations.
 
'''


class EnsemblePredictor(pl.LightningModule):
    def __init__(self, weightList: list, modelClass: pl.LightningModule):
        super(EnsemblePredictor, self).__init__()
        self.weightFiles = weightList
        self.models = [
            modelClass.load_from_checkpoint(weights).eval().cuda()
            for weights in self.weightFiles
        ]

    def forward(self, x):
        factors = list(range(1, 5))
        predictions = []
        dumb = self.models[0](x)
        vertFlip =  transforms.RandomVerticalFlip(p=1)
        horiFlip =  transforms.RandomHorizontalFlip(p=1)

        '''
        for f in factors:
            inp = x
            rot = torch.rot90(inp, f, dims=(2, 3))
            modelAvg = (sum([
                torch.rot90(m(rot), -f, dims=(2, 3))
                for  m in self.models
            ]) / len(self.models))
            predictions.append(modelAvg)
        final = sum(predictions) / len(predictions)
        '''

        inp = x
        vflip = vertFlip(inp)
        hflip = horiFlip(inp)

        modelAvg = (sum([
            vertFlip(m(vflip)) + horiFlip(m(hflip)) for m in self.models
            ]) / len(self.models))
        predictions.append(modelAvg)

        final = sum(predictions) / len(predictions)

        f, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(F.sigmoid(dumb).detach().cpu().numpy()[0][0],
                   cmap='gist_heat')
        ax2.imshow(F.sigmoid(final).detach().cpu().numpy()[0][0],
                   cmap='gist_heat')
        plt.show()

        return final
