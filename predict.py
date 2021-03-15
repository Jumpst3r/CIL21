import os
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split, TensorDataset
from pytorch_lightning.metrics import functional as FM
import numpy as np
import glob
import cv2
from PIL import Image, ImageOps
from pytorch_lightning.callbacks import ModelCheckpoint
from models.unet import UNet
import matplotlib.pyplot as plt
from post import crf

model = UNet.load_from_checkpoint('weights.ckpt')

# Iterate through a bunch of pictures in the test set

test_imgs = sorted(glob.glob('test_images/test_images/*.png'))
# test_imgs = sorted(glob.glob('training/training/images/*.png'))
for image_path in test_imgs:
    im = Image.open(image_path)
    im_org = Image.open(image_path)
    np_im = np.moveaxis(np.array(im),-1,0)
    np_im_org = np.array(im_org)
    model_in = torch.from_numpy(np_im).to(torch.float32).unsqueeze(0)
    model_in -= model_in.mean()
    model_in /= model_in.std()
    out = model(model_in)
    im = torch.round(F.sigmoid(out[0]))
    kernel = np.ones((5,5),np.uint8)
    dilation = cv2.dilate(im[0].detach().numpy(),kernel,iterations = 1)
    im2 = crf(np_im_org, torch.tensor(dilation).unsqueeze(0))
    f, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.imshow(np_im_org)
    ax2.imshow(im[0].detach().numpy(), cmap='binary_r')
    ax3.imshow(im2, cmap='binary_r')
    plt.show()