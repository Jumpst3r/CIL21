import os
# from comet_ml import Experiment
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image,ImageOps
import pytorch_lightning as pl
from torch.utils import data
from torch.utils.data import DataLoader, random_split, Dataset
from pytorch_lightning.metrics import functional as FM
import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from torchvision import transforms
import matplotlib.pyplot as plt
from models.unet import StackedUNet
# from models.unet.janikunet import StackedUNetPL
from tqdm import tqdm
from pytorch_lightning.loggers import CometLogger
import pytorch_lightning
from sklearn.model_selection import KFold
from dataset import ArealDataset, dataset_transform

from models.unet.utils import F1, accuracy

pl.seed_everything(2)

from sklearn.model_selection import KFold
import gc

if __name__ == '__main__':
    dataset = ArealDataset(root_dir_images='training/training/images/',root_dir_gt='training/training/groundtruth/', target_size=(256,256))
    # kf = KFold(n_splits=4)
    # val_IoU = []
    # val_F1 = []
    # val_acc = []
    #
    # for train_indices, test_indices in kf.split(dataset):
    #     train_dataset = torch.utils.data.dataset.Subset(dataset, train_indices)
    #     test_dataset = torch.utils.data.dataset.Subset(dataset, test_indices)
    #     test_dataset.applyTransforms = False
    #     train_dataloader = DataLoader(train_dataset, batch_size=5, pin_memory=True, num_workers=8)
    #     test_dataloader = DataLoader(test_dataset, batch_size=5, pin_memory=True, num_workers=8)
    #     model = StackedUNet(lr=1e-4, nb_blocks=4, share_weights=False, unet_mode='classic-backbone')
    #     trainer = pl.Trainer(max_epochs=150, gpus=1, stochastic_weight_avg=True, precision=16, deterministic=True, checkpoint_callback=False)
    #     trainer.fit(model, train_dataloader)
    #     results = trainer.test(model, test_dataloader, verbose=False)[0]
    #     val_IoU.append(results['results'][0])
    #     val_F1.append(results['results'][1])
    #     val_acc.append(results['results'][2])
    #     print('IoU = ', results['results'][0])
    #     print('F1 = ', results['results'][1])
    #     print('acc = ', results['results'][2])
    #     # print('Testing:')
    #     # with torch.no_grad():
    #     #     ious = []
    #     #     f1s = []
    #     #     accs = []
    #     #     model = model.cuda().eval()
    #     #     for batch in tqdm(test_dataloader):
    #     #         x, y = batch
    #     #         x, y = x.cuda(), y.cuda()
    #     #         out = model(x)
    #     #         iou = model.IoU(out, y)
    #     #         f = F1(out, y)
    #     #         acc = accuracy(out, y)
    #     #         ious.append(iou.detach().cpu().item())
    #     #         f1s.append(f.detach().cpu().item())
    #     #         accs.append(acc.detach().cpu().item())
    #     # mean_iou = np.array(ious).mean()
    #     # mean_f1 = np.array(f1s).mean()
    #     # mean_acc = np.array(accs).mean()
    #     # val_IoU.append(mean_iou)
    #     # val_F1.append(mean_f1)
    #     # val_acc.append(mean_acc)
    #     # print('IoU = ', mean_iou)
    #     # print('F1 = ', mean_f1)
    #     # print('acc = ', mean_acc)
    #     del model
    #     del trainer
    #     del train_dataset
    #     del test_dataset
    #     del train_dataloader
    #     del test_dataloader
    #     gc.collect()
    #     torch.cuda.empty_cache()
    #
    # print('mean/std IoU:', np.array(val_IoU).mean(), np.array(val_IoU).std())
    # print('mean/std F1:', np.array(val_F1).mean(), np.array(val_F1).std())
    # print('mean/std acc:', np.array(val_acc).mean(), np.array(val_acc).std())

    # train on full dataset to get kaggle score:
    num_samples_total = len(dataset)
    num_train = int(0.7 * num_samples_total)
    num_val = num_samples_total - num_train
    train, val = random_split(dataset, [num_train, num_val])


    train_dataloader = DataLoader(train, batch_size=10, num_workers=8)
    val_dataloader =  DataLoader(val, batch_size=5, num_workers=8)
    
    # for x,y in train_dataloader: vizualize(x,y)

    checkpoint_callback = ModelCheckpoint(
        dirpath='./',
        filename='weights',
        monitor='F1 val',
        mode='max',
        save_top_k=5
    )
    model = StackedUNet(lr=1e-4, nb_blocks=4, share_weights=False, unet_mode='classic-backbone')
    trainer = pl.Trainer(max_epochs=5000, gpus=1, stochastic_weight_avg=True, precision=16,
                         checkpoint_callback=checkpoint_callback)
    trainer.fit(model, train_dataloader, val_dataloader)
       
    

