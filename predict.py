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
from models.unet import UNet, StackedUNet
import matplotlib.pyplot as plt
from post import crf
from tqdm import tqdm
from torchvision import transforms
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from dataset_exploration import getnormvals

with torch.no_grad():

    model = StackedUNet.load_from_checkpoint('backbone-8/epoch=150-step=3000.ckpt').eval().cuda()

    # Iterate through a bunch of pictures in the test set

    os.makedirs('out', exist_ok=True)

    test_imgs = sorted(glob.glob('test_images/test_images/*.png'))
    # test_imgs = sorted(glob.glob('training/training/images/*.png'))
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
        # np_im = np.moveaxis(np.array(im, dtype=np.float32),-1,0)
        np_im_org = np.array(im_org, dtype=np.uint8)
        transform = A.Compose([
            A.Resize(SIZE, SIZE),
            A.Normalize(mean=means, std=stds),
            ToTensorV2(transpose_mask=True)
        ])
        tf = transform(image=im)
        # this transform doesn't include the norm.
        transform2 = A.Compose([
            A.Resize(SIZE, SIZE),
            ToTensorV2(transpose_mask=True)
        ])
        tf2 = transform2(image=im_org)

        im = tf['image'].unsqueeze(0).cuda()
        im_org = tf2['image'].unsqueeze(0)

        y = model(im)
        out = np.array(F.sigmoid(y[0]).detach().cpu().numpy(), dtype=np.float32)

        imout = np.array(out[0] > 0.8, dtype=np.float32)

        # im2 = crf(im[0], out)
        '''
        f, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(np.moveaxis(np.array(tf2['image']), 0,-1))
        ax2.imshow(imout, cmap='binary_r')
        plt.show()
        '''

        im = Image.fromarray(np.array(imout * 255, dtype=np.uint8)).resize((608, 608))
        im = im.resize((608, 608))
        fname = image_path[image_path.rfind('_') - 4:]
        im.save('out/' + fname)