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
from baseline_training import BaselineMLP
from feature_extractor import PATCH_SIZE
import matplotlib.pyplot as plt
model = BaselineMLP.load_from_checkpoint('weights-v7.ckpt')

# Iterate through a bunch of pictures in the test set

test_imgs = sorted(glob.glob('test_images/test_images/*.png'))
# test_imgs = glob.glob('training/training/images/*.png')
for image_path in test_imgs:
    im = Image.open(image_path)
    im_org = Image.open(image_path)
    np_im = np.array(im)
    np_im_org = np.array(im_org)
    scaled = cv2.resize(np_im, dsize=(400, 400), interpolation=cv2.INTER_CUBIC)
    scaled_org = cv2.resize(np_im_org, dsize=(400, 400), interpolation=cv2.INTER_CUBIC)
    opencvImage = scaled_org.copy()
    original = opencvImage.copy()
    np_im = scaled
    #plt.imshow(np_im)
    #plt.show()
    patches_per_row = (np_im.shape[0]) // PATCH_SIZE
    features = np.zeros(shape=(patches_per_row**2, 6))
    ft_idx = 0
    for y in range(0,patches_per_row*PATCH_SIZE,PATCH_SIZE):
        for x in range(0,patches_per_row*PATCH_SIZE,PATCH_SIZE):
                patch = np_im[y:y+PATCH_SIZE,x:x+PATCH_SIZE]
                means = [patch[:,:,c].mean() for c in range(3)]
                std = [patch[:,:,c].std() for c in range(3)]
                features[ft_idx] = np.concatenate((means,std))
                ft_idx += 1

    model_in = torch.from_numpy(features).to(torch.float32)
    print("MEAN: ",  model_in.mean())
    print("STD: ",  model_in.std())
    model_in -= model_in.mean()
    model_in /= model_in.std()
    out = model(model_in).view(-1,2)
    # Draw predictions on image :
    ft_idx = 0
    for y in range(0,patches_per_row*PATCH_SIZE,PATCH_SIZE):
        for x in range(0,patches_per_row*PATCH_SIZE,PATCH_SIZE):
                cv2.rectangle(opencvImage, (x, y), (x+PATCH_SIZE, y+PATCH_SIZE), (255,0,0) if out[ft_idx][1] > out[ft_idx][0] else (0,255,0,0.4), -1)
                ft_idx += 1

    res = cv2.addWeighted(original, 1.0, opencvImage, 0.25, 1)
    cv2.imshow('prediction', res)
    cv2.waitKey()
    # cv2.imshow('debug_image', opencvImage)
    # cv2.waitKey()