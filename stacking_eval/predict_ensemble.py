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
    pl.seed_everything(2)

    model_ckpts = glob.glob('ensemble_deepglobe/*/*.ckpt')
    models = [StackedUNet.load_from_checkpoint(ckpt).eval().cuda() for ckpt in model_ckpts]
    print(len(models))

    os.makedirs('out', exist_ok=True)

    test_imgs = sorted(glob.glob('test_images/test_images/*.png'))
    cnt = 0

    # The input size on which your model was trained
    SIZE = 480

    # means, stds = getnormvals()
    means, stds = np.load('means.npy'), np.load('stds.npy')

    print(means)
    print(stds)

    flip_dims = [[], [2], [3], [2, 3]]

    for image_path in tqdm(test_imgs):
        y_s = []
        for model in models:
            for fd in flip_dims:
                im = np.array(Image.open(image_path))
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

                im = tf['image'].unsqueeze(0).cuda()

                if len(fd) > 0:
                    im = torch.flip(im, fd)

                y = model(im)

                if len(fd) > 0:
                    y = torch.flip(y, fd)

                y_s.append(y.clone())

        y = torch.mean(torch.stack(y_s), dim=0)
        out = np.array(F.sigmoid(y[0]).detach().cpu().numpy(),
                       dtype=np.float32)

        imout = np.array(out[0] > 0.5, dtype=np.float32)

        im = Image.fromarray(np.array(imout * 255, dtype=np.uint8)).resize(
            (608, 608))
        im = im.resize((608, 608))
        fname = image_path[image_path.rfind('_') - 4:]
        im.save('out/' + fname)
