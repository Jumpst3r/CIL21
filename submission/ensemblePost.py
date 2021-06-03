from albumentations.augmentations.functional import rot90
import torch.nn.functional as F
import pytorch_lightning as pl
import albumentations as A
import torch
import numpy as np
from albumentations.pytorch.transforms import ToTensorV2
import matplotlib.pyplot as plt


'''
Two goals:

 - Implement voting style ensemble techniques
 - Implement test time augmentations.

'''

class EnsemblePredictor(pl.LightningModule):
    def __init__(self, weightList: list, modelClass: pl.LightningModule):
        super(EnsemblePredictor, self).__init__()
        self.weightFiles = weightList
        self.models = [modelClass.load_from_checkpoint(weights).eval().cuda() for weights in self.weightFiles]
        self.weights = [0.02,0.08,0.1, 0.3, 0.5]
    def forward(self, x):
        factors = list(range(1,5))
        predictions = []
        dumb = self.models[0](x)
        for f in factors:
            inp = x + torch.normal(mean = torch.zeros(x.shape), std=0.05).cuda()
            rot = torch.rot90(inp, f, dims=(2,3))
            modelAvg = (sum([torch.rot90(m(rot), -f, dims=(2,3)) for w,m in zip(self.weights,self.models)]) / len(self.models))
            predictions.append(modelAvg)
        final = sum(predictions) / len(predictions)
        
        f, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(F.sigmoid(dumb).detach().cpu().numpy()[0][0], cmap='gist_heat')
        ax2.imshow(F.sigmoid(final).detach().cpu().numpy()[0][0], cmap='gist_heat')
        plt.show()
        
        return final