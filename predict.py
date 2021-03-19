import glob
from tqdm import tqdm
from PIL import Image, ImageOps
import numpy as np
import torch
from models.UNet import UNet
import matplotlib.pyplot as plt
import os

model = UNet.load_from_checkpoint('weights-1.ckpt')
os.makedirs('out', exist_ok=True)

test_imgs = glob.glob('test_images/test_images/*.png')
for image_path in tqdm(test_imgs):
    im = Image.open(image_path)
    im_org = Image.open(image_path)
    np_im = np.moveaxis(np.array(im), -1, 0)
    np_im_org = np.array(im_org)
    model_in = torch.from_numpy(np_im/255.).to(torch.float32).unsqueeze(0)
    # model_in -= model_in.mean()
    # model_in /= model_in.std()
    out = model(model_in)
    # im = out
    im = torch.round(out)
    # im = torch.sigmoid(out[0])
    # im = torch.round(im)
    # im[im >= 0.1] = 1
    # im[im < 1] = 0
    # plt.imshow(im[0][0].detach().cpu().numpy())
    # TODO: opencv closing operation

    im_pil = Image.fromarray((im[0][0].detach().cpu().numpy()*255).astype(np.uint8))
    file_name = image_path.split('\\')[-1]
    print(file_name)
    im_pil.save('out/' + file_name)



