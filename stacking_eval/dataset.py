import glob

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import torch
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm


class ArealDataset(Dataset):
    """
    Dataset class with inbuild transformations based on the Dihedral group D4,
    according to: https://arxiv.org/pdf/1809.06839.pdf (p2 section B) these
    transformations work best.
    Dihedral groups: https://en.wikipedia.org/wiki/Dihedral_group

    params:
    root_dir_images: Root directory for input images (type png), must end with tailing /: String
    root_dir_gt: Root directory for binary masks (type png), must end with tailing /: String
    target_size: Resize target, default is (608,608): tuple
    visualize: Plot the images generated
    """
    def __init__(self,
                 root_dir_images: str,
                 root_dir_gt: str,
                 target_size=(100, 100),
                 visualize=False):
        impaths = sorted(glob.glob(root_dir_images + '*.png'))
        gtpaths = sorted(glob.glob(root_dir_gt + '*.png'))
        self.images = impaths
        self.gt = gtpaths
        self.visualize = visualize
        # Get dataset mean and stds:
        means = []
        stds = []
        if len(means) == 0:
            count = 0
            for imname in tqdm(impaths):
                pil_im = np.array(Image.open(imname))
                pil_im = pil_im / 255  # normalize betweeen 0 1
                means.append([pil_im[:, :, c].mean() for c in range(3)])
                stds.append([pil_im[:, :, c].std() for c in range(3)])
                count += 1
            means = list(np.array(means).sum(axis=0) / count)
            stds = list(np.array(stds).sum(axis=0) / count)
        # means, stds = list(np.load('means.npy')), list(np.load('stds.npy'))
        print(means, stds)
        self.transform_train = A.Compose([
            A.RandomResizedCrop(height=target_size[0], width=target_size[1], scale=(0.5, 1.0), ratio=(1.0, 1.0)),
            A.ColorJitter(),
            A.RandomRotate90(),
            A.VerticalFlip(),
            A.HorizontalFlip(),
            A.Transpose(),
            A.Normalize(mean=means, std=stds),
            ToTensorV2(transpose_mask=True)
        ])
        self.transform_test = A.Compose([
            A.Resize(target_size[0], target_size[1]),
            A.Normalize(mean=means, std=stds),
            ToTensorV2(transpose_mask=True)
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        impath = self.images[idx]
        gtpath = self.gt[idx]
        image = np.array(Image.open(impath))
        gt = np.array(Image.open(gtpath))
        gt = np.array(gt > 100, dtype=np.float32)
        if self.applyTransforms:
            tf = self.transform_train(image=image, mask=gt)
        else:
            tf = self.transform_test(image=image, mask=gt)
        im = tf['image']
        gt = tf['mask']
        if self.visualize:
            f, (ax1, ax2) = plt.subplots(1, 2)
            ax1.axis('off')
            ax2.axis('off')
            ax1.imshow(np.moveaxis(np.array(im), 0, -1))
            ax2.imshow(gt, cmap='binary_r')
            plt.show()
        # use with cat. cross-entropy
        gt = tf['mask'].unsqueeze(0)
        return im.type(torch.float32), gt.type(torch.float32)
