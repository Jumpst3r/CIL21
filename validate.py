import glob
from tqdm import tqdm
from PIL import Image, ImageOps
import numpy as np
import torch
from models.StackedUNet import UNet
import matplotlib.pyplot as plt
import os
import cv2
from postprocessing.crf import dense_crf
from helpers import get_accuracy

# with torch.no_grad():

erode = False
weight_file = r'weights/weights_stacks-4_bs-1_imgsize-400_trainsets-cil_lr-0.0001_savelast-True_norm-False.ckpt'
model = UNet.load_from_checkpoint(weight_file)
img_size = int(weight_file.split('imgsize-')[-1].split('_')[0])

# turn off gradient computation, batchnorm, dropout
torch.set_grad_enabled(False)
model.eval()

val_imgs = glob.glob(r'training/validation/images/*.png')
accuracies = []
for image_path in tqdm(val_imgs):

    # load image
    im = Image.open(image_path)
    orig_size = im.size
    im = im.resize((img_size, img_size))
    np_im = np.moveaxis(np.array(im), -1, 0)
    model_in_img = torch.from_numpy(np_im/255.).to(torch.float32)

    # load ground truth
    gt = Image.open(image_path.replace('images', 'groundtruth'))
    gt = np.round(np.array(gt)/255.)

    if 'norm-True' in weight_file:
        model_in_img -= model_in_img.mean()
        model_in_img /= model_in_img.std()

    # make prediction
    model_in = model_in_img
    preds = model(model_in.unsqueeze(0))
    out = preds[-1]

    # threshold
    im = out[0][0].detach().cpu().numpy().copy()
    im[im > 0.8] = 1
    im[im < 1] = 0

    # TODO: also test if stuff like this is beneficial for accuracy on the validation set
    if erode:
        opening = cv2.morphologyEx(im, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
        erosion = cv2.erode(opening, np.ones((7, 7), np.uint8), iterations=1)
        final_pred = erosion
    else:
        final_pred = im

    # resize to original size
    final_pred = cv2.resize(final_pred, orig_size, interpolation=cv2.INTER_NEAREST)

    acc = get_accuracy(final_pred, gt)
    # print(acc)
    accuracies.append(acc)

mean_acc = sum(accuracies) / len(accuracies)
print('Validation accuracy = ', round(mean_acc, 5))
