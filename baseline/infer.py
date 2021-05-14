import os
import numpy as np
import matplotlib.image as mpimg
import re

# ==============
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
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from baseline_models import VisionBaseline
from torchvision.models.segmentation import fcn_resnet50, fcn_resnet101, deeplabv3_resnet50, deeplabv3_resnet101
import argparse
import time
import cv2 as cv


import glob

foreground_threshold = 0.5 # percentage of pixels > 1 required to assign a foreground label to a patch

# assign a label to a patch
def patch_to_label(patch):
    df = np.mean(patch)
    if df > foreground_threshold:
        return 1
    else:
        return 0


def mask_to_submission_strings(image_filename):
    """Reads a single image and outputs the strings that should go into the submission file"""
    img_number = int(re.search(r"\d+.png", image_filename).group(0)[:-4]) #crop .png
    im = mpimg.imread(image_filename)
    patch_size = 16
    for j in range(0, im.shape[1], patch_size):
        for i in range(0, im.shape[0], patch_size):
            patch = im[i:i + patch_size, j:j + patch_size]
            label = patch_to_label(patch)
            yield("{:03d}_{}_{},{}".format(img_number, j, i, label))


def masks_to_submission(submission_filename, *image_filenames):
    """Converts images into a submission file"""
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for fn in image_filenames[0:]:
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(fn))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("key", help="model which should be trained", type=str)
    parser.add_argument("mode", help="native or cropped inference style", type=str)
    key = vars(parser.parse_args())['key']
    mode = vars(parser.parse_args())['mode']

    seg_models = {"fcn_resnet50": fcn_resnet50, "fcn_resnet101": fcn_resnet101,
                  "deeplabv3_resnet50": deeplabv3_resnet50, "deeplabv3_resnet101": deeplabv3_resnet101}
    model_options = dict(pretrained=False, progress=True, num_classes=1)
    optimizer = torch.optim.Adam
    base_adam_options = dict(lr=1e-4, weight_decay=1e-5)

    test_imgs = sorted(glob.glob('test_images/*.png'))

    def getNorms():
        means = []
        stds =  []
        if len(means) == 0:
            count = 0
            d = 1/255
            for imname in tqdm(test_imgs):
                pil_im = np.array(Image.open(imname))
                pil_im = pil_im * d# normalize betweeen 0 1
                means.append([pil_im[:,:,c].mean() for c in range(3)])
                stds.append([pil_im[:,:,c].std() for c in range(3)])
                count += 1
            means = list(np.array(means).sum(axis=0) / count)
            stds = list(np.array(stds).sum(axis=0) / count)
        return means, stds

    def getModel(key):
        path = "./trained_models/" + key + "_trained.pt"
        model = VisionBaseline(seg_models[key], model_options, F.binary_cross_entropy_with_logits, optimizer,
                               base_adam_options, 100)
        model.load_state_dict(torch.load(path))
        model.eval()
        return model

    def inferNativeRes(key):
        model = getModel(key)

        test_imgs = sorted(glob.glob('test_images/*.png'))
        cnt = 0
        # The input size on which your model was trained
        new_size = 389 #194
        ref = 256 #128
        diff = new_size - ref

        means, stds = getNorms()
        print(means, stds)

        c0 = (0, 0, 400, 400)
        c1 = (208, 0, 608, 400)
        c2 = (0, 208, 400, 608)
        c3 = (208, 208, 608, 608)
        crops = [c0, c1, c2, c3]

        # define matrix used to calculate mean of re-combined samples
        mean_mat = np.ones((new_size, new_size))
        mean_mat[diff:ref, diff:ref] = 0.25
        mean_mat[:diff, diff:ref] = 0.5
        mean_mat[ref:, diff:ref] = 0.5
        mean_mat[diff:ref, :diff] = 0.5
        mean_mat[diff:ref, ref:] = 0.5

        for image_path in tqdm(test_imgs):
            cnt += 1
            im = Image.open(image_path)

            transform = A.Compose([
                A.Resize(ref, ref),
                A.Normalize(mean=means, std=stds),
                ToTensorV2(transpose_mask=True)
            ])

            tf = torch.empty(4, 3, ref, ref).float()
            for i, c in enumerate(crops):
                im_c = np.array(im.crop(c))
                im_c = transform(image=im_c)
                im_c = im_c['image'].unsqueeze(0)
                tf[i, :, :, :] = im_c
                
            y = model(tf) #shape: (4, 1, 128, 128)
            y = np.array(y.detach().cpu().numpy(), dtype=np.float32)

            comb = np.zeros((new_size, new_size))
            comb[:ref, :ref] += y[0, 0, :, :]
            comb[:ref, diff:] += y[1, 0, :, :]
            comb[diff:, :ref] += y[2, 0, :, :]
            comb[diff:, diff:] += y[3, 0, :, :]

            comb *= mean_mat

            out = np.array(F.sigmoid(torch.tensor(comb)).detach().cpu().numpy(), dtype=np.float32)
            #out = np.array(out > 0.5, dtype=np.float32)

            out = np.array(out * 255, dtype=np.uint8)

            out = cv.adaptiveThreshold(out, 1, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 0)
            im = Image.fromarray(out).resize((608, 608))
            #im = im.resize((608, 608))
            fname = image_path[image_path.rfind('_') - 4:]
            im.save('./out_'+key+'_native_thresh/' + fname)


    def infer(key):
        model = getModel(key)

        test_imgs = sorted(glob.glob('test_images/*.png'))
        cnt = 0
        # The input size on which your model was trained
        SIZE = 128

        means, stds = getNorms()
        print(means, stds)

        for image_path in tqdm(test_imgs):
            #print("processing img nr: ", cnt)
            cnt += 1
            im = np.array(Image.open(image_path))
            transform = A.Compose([
                A.Resize(SIZE, SIZE),
                A.Normalize(mean=means, std=stds),
                ToTensorV2(transpose_mask=True)
            ])
            tf = transform(image=im)

            im = tf['image'].unsqueeze(0)

            y = model(im)
            out = np.array(F.sigmoid(y[0]).detach().cpu().numpy(), dtype=np.float32)

            imout = np.array(out[0] > 0.5, dtype=np.float32)

            '''
            f, (ax1, ax2) = plt.subplots(1, 2)
            ax1.imshow(np.moveaxis(np.array(tf2['image']), 0,-1))
            ax2.imshow(imout, cmap='binary_r')
            plt.show()
            '''

            im = Image.fromarray(np.array(imout * 255, dtype=np.uint8)).resize((608, 608))
            im = im.resize((608, 608))
            fname = image_path[image_path.rfind('_') - 4:]
            im.save('./out_'+key+'/' + fname)

    if mode == "native":
        inferNativeRes(key)
        submission_filename = key + '_native_test_submission.csv'
        image_filenames = glob.glob('./out_' + key + '_native_thresh/*.png')
    if mode == "crop":
        infer(key)
        submission_filename = key + 'test_submission.csv'
        image_filenames = glob.glob('./out_' + key + '/*.png')

    masks_to_submission(submission_filename, *image_filenames)
    print("done! File: ", submission_filename)


