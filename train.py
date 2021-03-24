import os
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split, Dataset
from pytorch_lightning.metrics import functional as FM
import numpy as np
import glob
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data.sampler import WeightedRandomSampler
import matplotlib.pyplot as plt
from torchvision import transforms
import matplotlib.pyplot as plt
from models.unet import UNet
from tqdm import tqdm
cnt = 0

class RoadDataset(Dataset):

    def __init__(self, root_dir_images, root_dir_gt, transform=None):
        self.images = sorted(glob.glob(root_dir_images + '*.png'))
        self.gt = sorted(glob.glob(root_dir_gt + '*.png'))
        self.transform = transform


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        impath = self.images[idx]
        gtpath = self.gt[idx]
        pil_im = Image.open(impath)
        pil_gt = Image.open(gtpath)
        image =  transforms.ToTensor()(pil_im)
        gt = torch.tensor(np.where(np.array(pil_gt) >=  1., 1., 0.))
        image -= image.mean()
        image /= image.std()

        stacked = torch.zeros((4, image.shape[1], image.shape[2]))
        stacked[0] = image[0]
        stacked[1] = image[1]
        stacked[2] = image[2]
        stacked[3] = gt

        if self.transform is not None:
            stack_transformed = self.transform(stacked)
            image = stack_transformed[0:3]
            gt = stack_transformed[3]


        return image, gt.unsqueeze(0)

def vizualize(x,y):
    """Visualize a tensor pair, where x is a tensor of shape [1,3,width,height] and y is a tensor of shape [1,width, height]
    """    
    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.axis('off')
    ax2.axis('off')
    ax1.imshow(np.moveaxis(np.array(x[0]),0,-1))
    ax2.imshow(y[0][0], cmap='binary_r')
    plt.show()


def dataset_transform():
    global cnt
    images_names = sorted(glob.glob('/home/nicolas/new-dataset/test/sat/*.png'))
    gt_name = sorted(glob.glob('/home/nicolas/new-dataset/test/map/*.png'))

    for _ in tqdm(range(50)):
        for im_n, gt_n in zip(images_names, gt_name):
            pil_im = Image.open(im_n)
            pil_gt = Image.open(gt_n)
            image =  transforms.ToTensor()(pil_im)
            gt = torch.tensor(np.where(np.array(pil_gt) >=  1, 1., 0.)).unsqueeze(0)
            transform = transforms.RandomResizedCrop(400, scale=(0.02, 0.02), ratio=(1.,1.))
            state = torch.get_rng_state()
            image = transform(image)
            torch.set_rng_state(state)
            gt = transform(gt)
            if gt.mean() < 0.04: continue
            gt = np.array(gt*255, dtype=np.uint8)[0]
            im = Image.fromarray(np.array(gt, dtype=np.uint8))
            global cnt
            im.save(f'training/training/groundtruth/custom-{cnt}-mask.png')
            npim = np.moveaxis(np.array(image*255, dtype=np.uint8), 0, -1)
            im = Image.fromarray(npim)
            im.save(f'training/training/images/custom-{cnt}-input.png')
            cnt += 1



if __name__ == '__main__':
    # dataset_transform()
    my_transforms = transforms.Compose([
        transforms.RandomAffine(90, translate=[0.1,0.2], scale=[0.8,1.5], shear=2),
        #transforms.RandomResizedCrop(400, scale=(0.8, 1.0), ratio=(1., 1.), interpolation=Image.NEAREST),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomVerticalFlip(p=0.5)
    ])

    dataset = RoadDataset(root_dir_images='training/training/images/',root_dir_gt='training/training/groundtruth/', transform=my_transforms)
    num_samples_total = len(dataset)
    num_train = int(0.7 * num_samples_total)
    num_val = num_samples_total - num_train
    train, val = random_split(dataset, [num_train, num_val])


    train_dataloader = DataLoader(train, batch_size=5)
    val_dataloader =  DataLoader(val, batch_size=5)
    
    # for x,y in train_dataloader: vizualize(x,y)

    checkpoint_callback = ModelCheckpoint(
        dirpath='./',
        filename='weights',
        monitor='IoU val',
        mode='max'
    )
    fcn = UNet.load_from_checkpoint('weights-v7.ckpt')
    trainer = pl.Trainer(checkpoint_callback=checkpoint_callback, max_epochs=1000, gpus=1)
    trainer.fit(fcn,train_dataloader,val_dataloader)
