from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms, utils
from PIL import Image

"""
NOT USED
"""

class KaggleSet(Dataset):
    # official CIL21 training set
    def __init__(self, root_dir, idx_list, transform=None):
        self.root_dir = root_dir
        self.idx_list = idx_list
        self.transform = transform

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, idx):
        suffix = "_"
        idx = self.idx_list[idx]
        if idx < 10:
            suffix += "0"
        if idx < 100:
            suffix += "0"
        suffix += str(idx)+".png"

        img_name = self.root_dir + "/images/satImage" + suffix
        lbl_name = self.root_dir + "/groundtruth/satImage" + suffix

        img = Image.open(img_name)
        lbl = Image.open(lbl_name)

        img = transforms.ToTensor()(img)
        lbl = transforms.ToTensor()(lbl)

        # binarize the label image
        lbl = (lbl > 0.1).float()

        if self.transform is not None:
            img = self.transform(img)

        #img -= img.mean()
        #img /= img.std()

        #sample = torch.cat((img, lbl))

        return img, lbl

