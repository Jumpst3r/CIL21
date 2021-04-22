import glob
from tqdm import tqdm
from PIL import Image, ImageOps
import numpy as np
import torch
from models.StackedPatchNetV2 import PatchNet
import matplotlib.pyplot as plt
import os
import cv2
from postprocessing.crf import dense_crf
from helpers import get_accuracy
from torchvision import transforms

os.makedirs('out', exist_ok=True)
val_imgs = glob.glob(r'training_patch/validation/images/*.png')

weight_file = r'weights/patchnetv2_stack-5_bs-4_imgsize-608_trainsets-cil_lr-0.0001_dropout-False_ARCHTEST.ckpt'
model = PatchNet.load_from_checkpoint(weight_file).cuda()

# turn off gradient computation, batchnorm, dropout
torch.set_grad_enabled(False)
model.eval()

accuracies = []
for image_path in tqdm(val_imgs):

    # load image
    im = Image.open(image_path)
    im = im.resize((608, 608))
    # assert (orig_size == (608, 608))
    # np_im = np.moveaxis(np.array(im), -1, 0)
    # model_in_img = torch.from_numpy(np_im / 255.).to(torch.float32)
    model_in_img = transforms.ToTensor()(im)
    model_in_img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(model_in_img)

    # load ground truth
    gt = Image.open(image_path.replace('images', 'groundtruth_full'))
    gt = np.round(np.array(gt) / 255.)

    # make prediction
    model_in = model_in_img.cuda()
    pred = model(model_in.unsqueeze(0))[-1]

    # threshold
    thresh = 0.5
    im = pred.clone()
    im[im > thresh] = 1
    im[im < 1] = 0

    # upsample
    im_up = torch.nn.functional.interpolate(im, mode='nearest', size=(608, 608))
    im = im_up[0][0].detach().cpu().numpy().copy()
    final_pred = (im * 255).astype(np.uint8)

    acc = get_accuracy(final_pred, gt)
    print(acc)
    accuracies.append(acc)

mean_acc = sum(accuracies) / len(accuracies)
print('Validation accuracy = ', round(mean_acc, 5))