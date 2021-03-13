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
from baseline_training import FCN
import matplotlib.pyplot as plt
model = FCN.load_from_checkpoint('weights.ckpt')

# Iterate through a bunch of pictures in the test set

#test_imgs = sorted(glob.glob('test_images/test_images/*.png'))
test_imgs = glob.glob('training/training/images/*.png')
for image_path in test_imgs:
    im = Image.open(image_path)
    im_org = Image.open(image_path)
    np_im = np.array(im)
    np_im_org = np.array(im_org)
    scaled = cv2.resize(np_im, dsize=(400, 400), interpolation=cv2.INTER_LANCZOS4)
    scaled_org = cv2.resize(np_im_org, dsize=(400, 400), interpolation=cv2.INTER_LANCZOS4)
    opencvImage = scaled_org.copy()
    original = opencvImage.copy()
    np_im = np.moveaxis(scaled,-1,0)
    model_in = torch.from_numpy(np_im).to(torch.float32).unsqueeze(0)
    model_in -= model_in.mean()
    model_in /= model_in.std()
    out = model(model_in)
    im = torch.round(F.sigmoid(out[0]))
    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(opencvImage)
    ax2.imshow(im[0].detach().numpy(), cmap='binary_r')
    plt.show()