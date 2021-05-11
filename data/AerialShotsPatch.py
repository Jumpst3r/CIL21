import torch
import glob
from PIL import Image
from torchvision import transforms
# import matplotlib.pyplot as plt
import cv2
import numpy as np
from random import randrange


class AerialShotsPatch(torch.utils.data.Dataset):

    def __init__(self, data_paths, augment, img_size, val=False):
        self.data_paths = data_paths
        self.augment = augment

        self.img_size = img_size
        self.val = val

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.color_transforms = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        self.spatial_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=(-180, 180), interpolation=transforms.InterpolationMode.NEAREST),
        ])

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):

        assert (self.data_paths[idx][0].split('/')[-1].split('\\')[-1] ==
                self.data_paths[idx][1].split('/')[-1].split('\\')[-1])

        # load image & label
        img = Image.open(self.data_paths[idx][0])
        label = Image.open(self.data_paths[idx][1])

        img = img.resize((self.img_size, self.img_size))
        label = label.resize((self.img_size, self.img_size), resample=Image.NEAREST)

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

        # resize label to a 16th of the input resolution
        if not self.val:
            label = torch.nn.functional.interpolate(label.unsqueeze(1), mode='nearest', scale_factor=(1/16, 1/16))[0]
        else:
            label = torch.nn.functional.interpolate(label.unsqueeze(1), mode='nearest', size=(38, 38))[0]

        # normalize image
        img = self.normalize(img)

        return img, label



