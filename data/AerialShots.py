import torch
import glob
from PIL import Image
from torchvision import transforms
# import matplotlib.pyplot as plt
import cv2
import numpy as np

class AerialShots(torch.utils.data.Dataset):

    def __init__(self, data_paths, augment):
        self.data_paths = data_paths
        self.augment = augment

        self.img_size = 608

        self.color_transforms = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        self.spatial_transforms = transforms.Compose([
            # transforms.ToPILImage(),
            #transforms.RandomResizedCrop(size=self.img_size, scale=(0.5, 1.0), ratio=(1.0, 1.0), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=(-90, 90), interpolation=transforms.InterpolationMode.NEAREST),
            # transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):

        # load image & label
        img = Image.open(self.data_paths[idx][0])
        label = Image.open(self.data_paths[idx][1])

        img = img.resize((self.img_size, self.img_size))
        label = label.resize((self.img_size, self.img_size), resample=Image.NEAREST)

        # apply color transformations to image
        if self.augment:
            img = self.color_transforms(img)

        # # detect lines
        # gray = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
        # gray_blur = cv2.blur(gray, (5, 5))
        # edges = cv2.Canny(gray_blur, 50, 150, apertureSize=3)
        # lines = cv2.HoughLinesP(edges, rho=0.7, theta=np.pi / 180, threshold=70, minLineLength=20, maxLineGap=100)
        # line_img = np.zeros_like(img)
        # if lines is not None:
        #     lines = lines[:, 0, :]
        #     for x1, y1, x2, y2 in lines:
        #         cv2.line(line_img, (x1, y1), (x2, y2), (255,255,255), 2)
        # line_img = line_img[:, :, 0]

        # transform image to tensor
        img = transforms.ToTensor()(img)
        label = transforms.ToTensor()(label)
        # line_img = transforms.ToTensor()(line_img)

        # binarize the label image
        label = (label > 0.1).float()

        # apply spatial transformations to image and label
        # stacked = torch.zeros((5, self.img_size, self.img_size))
        stacked = torch.zeros((4, self.img_size, self.img_size))
        stacked[:3] = img
        # stacked[-2] = line_img
        stacked[-1] = label
        if self.augment:
            stacked = self.spatial_transforms(stacked)
        img = stacked[:3]
        # img = stacked[:4]
        label = stacked[-1].unsqueeze(0)

        # plt.imshow(img.numpy().transpose(1, 2, 0))
        # plt.figure()
        # plt.imshow(label[0].numpy())

        # binarize the label image again, just to make sure
        label = (label > 0.1).float()

        # # normalize image (TODO: try)
        img -= img.mean()
        img /= img.std()

        return img, label



