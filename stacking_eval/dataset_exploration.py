'''
This file can be used to gather statistics about the dataset (per channel means and stds)
'''

import glob
import numpy as np
from PIL import Image



def getnormvals():
    images_names = sorted(glob.glob('training/training/images/*.png'))

    means = []
    stds = []
    count = 0
    for imname in images_names:
        pil_im = np.array(Image.open(imname))
        pil_im = pil_im / 255  # normalize betweeen 0 1
        means.append([pil_im[:, :, c].mean() for c in range(3)])
        stds.append([pil_im[:, :, c].std() for c in range(3)])
        count += 1

    return (np.array(means).sum(axis=0) / count), (np.array(stds).sum(axis=0) /
                                                   count)
