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

# with torch.no_grad():

erode = False
weight_file = r'weights/weights_stack-8_bs-4_imgsize-160_trainsets-cil_lr-0.0001_dropout-False_norm-False_0.25.ckpt'
model = UNet.load_from_checkpoint(weight_file)
os.makedirs('out', exist_ok=True)
img_size = int(weight_file.split('imgsize-')[-1].split('_')[0])

# turn off gradient computation, batchnorm, dropout
torch.set_grad_enabled(False)
model.eval()

test_imgs = glob.glob(r'test_images/test_images/*.png')
for image_path in tqdm(test_imgs):

    # load image
    im = Image.open(image_path)
    orig_size = im.size
    im = im.resize((img_size, img_size))  # we always predict at training resolution! to not mess up learned scale
    np_im = np.moveaxis(np.array(im), -1, 0)
    model_in_img = torch.from_numpy(np_im/255.).to(torch.float32)

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
        final_pred = (erosion * 255).astype(np.uint8)
    else:
        final_pred = (im * 255).astype(np.uint8)

    # resize to original size
    final_pred = cv2.resize(final_pred, orig_size, interpolation=cv2.INTER_NEAREST)

    # write mask to disk
    im_pil = Image.fromarray(final_pred)
    file_name = image_path.split('/')[-1].split('\\')[-1]
    im_pil.save('out/' + file_name)







