import glob
import os

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image, ImageOps
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.metrics import functional as FM
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import transforms
# from post import crf
from tqdm import tqdm

from dataset_exploration import getnormvals
from models.unet import StackedUNet, UNet

with torch.no_grad():

    model = StackedUNet.load_from_checkpoint(
        'checkpoints/cb_6/epoch=150-step=3000.ckpt').eval().cuda()

    # Iterate through a bunch of pictures in the test set

    os.makedirs('out', exist_ok=True)

    test_imgs = sorted(glob.glob('test_images/test_images/*.png'))
    cnt = 0

    # The input size on which your model was trained
    SIZE = 192

    means, stds = getnormvals()

    print(means)
    print(stds)

    for image_path in tqdm(test_imgs):
        cnt += 1
        im = np.array(Image.open(image_path))
        im_org = np.array(Image.open(image_path).resize((SIZE, SIZE)))
        np_im_org = np.array(im_org, dtype=np.uint8)
        transform = A.Compose([
            A.Resize(SIZE, SIZE),
            A.Normalize(mean=means, std=stds),
            ToTensorV2(transpose_mask=True)
        ])
        tf = transform(image=im)
        # this transform doesn't include the norm.
        transform2 = A.Compose(
            [A.Resize(SIZE, SIZE),
             ToTensorV2(transpose_mask=True)])
        tf2 = transform2(image=im_org)

        im = tf['image'].unsqueeze(0).cuda()
        im_org = tf2['image'].unsqueeze(0)

        y = model(im)
        out = np.array(F.sigmoid(y[0]).detach().cpu().numpy(),
                       dtype=np.float32)

        imout = np.array(out[0] > 0.5, dtype=np.float32)

        im = Image.fromarray(np.array(imout * 255, dtype=np.uint8)).resize(
            (608, 608))
        im = im.resize((608, 608))
        fname = image_path[image_path.rfind('_') - 4:]
        im.save('out/' + fname)
