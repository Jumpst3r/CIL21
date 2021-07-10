import glob
import os

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from albumentations.augmentations.transforms import CLAHE, ColorJitter
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image, ImageOps
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.metrics import functional as FM
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import transforms
from tqdm import tqdm

from dataset_exploration import getnormvals
from ensemblePost import EnsemblePredictor
from models.unet import StackedUNet, UNet

with torch.no_grad():

    # model = StackedUNet.load_from_checkpoint('weights.ckpt').eval().cuda()
    # model = EnsemblePredictor(['weights/weights.ckpt','weights/weights-v1.ckpt' ,'weights/weights-v2.ckpt' ,'weights/weights-v3.ckpt', 'weights/weights-v4.ckpt'], StackedUNet).eval()
    model = EnsemblePredictor([
        'weights-v3.ckpt', 'weights-v4.ckpt', 'weights-v5.ckpt',
        'weights-v6.ckpt', 'weights-v7.ckpt'
    ], StackedUNet).eval()

    # Iterate through a bunch of pictures in the test set

    os.makedirs('out', exist_ok=True)

    test_imgs = sorted(glob.glob('test_images/test_images/*.png'))
    # test_imgs = sorted(glob.glob('training/training/images/*.png'))
    cnt = 0

    # The input size on which your model was trained
    SIZE = 480

    means, stds = getnormvals()

    print(means)
    print(stds)

    for image_path in tqdm(test_imgs[9:]):
        cnt += 1
        im = np.array(Image.open(image_path), dtype=np.uint8)
        im_org = np.array(Image.open(image_path).resize((SIZE, SIZE)),
                          dtype=np.uint8)
        # np_im = np.moveaxis(np.array(im, dtype=np.float32),-1,0)
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

        imout = np.array(out[0] > 0.8, dtype=np.float32)

        # im2 = crf(im[0], out)

        f, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(np.moveaxis(np.array(tf2['image']), 0, -1))
        ax2.imshow(imout, cmap='binary_r')
        plt.show()

        im = Image.fromarray(np.array(imout * 255, dtype=np.uint8)).resize(
            (608, 608))
        im = im.resize((608, 608))
        fname = image_path[image_path.rfind('_') - 4:]
        im.save('out/' + fname)
