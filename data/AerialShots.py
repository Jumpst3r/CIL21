import torch
import glob
from PIL import Image
from torchvision import transforms
# import matplotlib.pyplot as plt
import cv2
import numpy as np
from random import randrange

class AerialShots(torch.utils.data.Dataset):

    def __init__(self, data_paths, augment, normalize, img_size=400, label_size=400):
        self.data_paths = data_paths
        self.augment = augment

        self.img_size = img_size
        self.label_size = label_size
        self.normalize = normalize

        self.color_transforms = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        self.spatial_transforms = transforms.Compose([
            # TODO: try
            #transforms.RandomResizedCrop(size=self.img_size, scale=(0.5, 1.0), ratio=(1.0, 1.0), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=(-90, 90), interpolation=transforms.InterpolationMode.NEAREST),
        ])

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):

        assert (self.data_paths[idx][0].split('/')[-1].split('\\')[-1] ==
                self.data_paths[idx][1].split('/')[-1].split('\\')[-1])

        # load image & label
        img = Image.open(self.data_paths[idx][0])
        label = Image.open(self.data_paths[idx][1])

        # if img is larger than 1000x1000, take a random 400x400 crop
        crop_size = 200
        if max(img.size) > 1000:
            x1 = randrange(0, img.size[1] - crop_size)
            y1 = randrange(0, img.size[0] - crop_size)
            img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
            label = label.crop((x1, y1, x1 + crop_size, y1 + crop_size))

        img = img.resize((self.img_size, self.img_size))
        label = label.resize((self.label_size, self.label_size), resample=Image.NEAREST)

        # apply color transformations to image
        if self.augment:
            img = self.color_transforms(img)

        # transform image to tensor
        img = transforms.ToTensor()(img)
        label = transforms.ToTensor()(label)

        # binarize the label image
        label = (label > 0.1).float()

        # apply spatial transformations to image and label
        if self.augment:
            stacked = torch.zeros((4, self.img_size, self.img_size))
            stacked[:3] = img
            stacked[-1] = label
            stacked = self.spatial_transforms(stacked)
            img = stacked[:3]
            label = stacked[-1].unsqueeze(0)

        # binarize the label image again, just to make sure
        label = (label > 0.1).float()

        # normalize image
        if self.normalize:
            img -= img.mean()
            img /= img.std()

        return img, label



