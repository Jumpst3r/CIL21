from PIL import Image
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset
import torch
import matplotlib.pyplot as plt
import glob
import numpy as np
import os


class ArealDatasetTest(Dataset):
    def __init__(self, root_dir_images, target_size=(608,608), means=None, stds=None):
        impaths = sorted(glob.glob(root_dir_images + '*.png'))
        self.names = [os.path.basename(path) for path in impaths]
        self.images = impaths
        self.means = means
        self.stds = stds

        self.transform = A.Compose([
            A.Resize(target_size[0], target_size[1]),
            A.Normalize(mean=self.means, std=self.stds),
            ToTensorV2(transpose_mask=True)
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        impath = self.images[idx]
        image = np.array(Image.open(impath))
        tf = self.transform(image=image)
        im = tf['image']

        return im.type(torch.float32), self.names[idx]