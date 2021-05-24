from PIL import Image
import albumentations as A
from albumentations.augmentations.transforms import CLAHE, ColorJitter
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset
import torch
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torchvision import transforms

class ArealDataset(Dataset):
    '''
    Dataset class with inbuild transformations based on the Dihedral group D4,
    according to: https://arxiv.org/pdf/1809.06839.pdf (p2 section B) these
    transformations work best.
    Dihedral groups: https://en.wikipedia.org/wiki/Dihedral_group

    params:
    root_dir_images: Root directory for input images (type png), must end with tailing /: String
    root_dir_gt: Root directory for binary masks (type png), must end with tailing /: String
    target_size: Resize target, default is (608,608): tuple
    visualize: Plot the images generated
    '''
    def __init__(self, root_dir_images: str, root_dir_gt: str, target_size=(100,100), visualize=False, applyTransforms=True):
        impaths = sorted(glob.glob(root_dir_images + '*.png'))
        gtpaths = sorted(glob.glob(root_dir_gt + '*.png'))
        self.images = impaths
        self.gt = gtpaths
        self.visualize = visualize
        # Get dataset mean and stds:
        # 
        means = [] 
        stds =  [] 
        if len(means) == 0:
            count = 0
            for imname in tqdm(impaths):
                pil_im = np.array(Image.open(imname))
                pil_im = pil_im / 255 # normalize betweeen 0 1
                means.append([pil_im[:,:,c].mean() for c in range(3)])
                stds.append([pil_im[:,:,c].std() for c in range(3)])
                count += 1
            means = list(np.array(means).sum(axis=0) / count)
            stds = list(np.array(stds).sum(axis=0) / count)
        print(means, stds)
        if applyTransforms:
            self.transform = A.Compose([
                A.Resize(target_size[0], target_size[1]),
                A.RandomRotate90(),
                A.VerticalFlip(),
                A.HorizontalFlip(),
                A.Transpose(),
                A.ColorJitter(),
                A.Normalize(mean=means, std=stds),
                ToTensorV2(transpose_mask=True)
            ])
        else:
            self.transform = A.Compose([
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
        tf = self.transform(image=image, mask=gt)
        im = tf['image']
        gt = tf['mask']
        if self.visualize:
                f, (ax1, ax2) = plt.subplots(1, 2)
                ax1.axis('off')
                ax2.axis('off')
                ax1.imshow(np.moveaxis(np.array(im),0,-1))
                ax2.imshow(gt, cmap='binary_r')
                plt.show()
        # use with cat. cross-entropy
        gt = tf['mask'].unsqueeze(0)
        return im.type(torch.float32), gt.type(torch.float32)  

def dataset_transform():
   
    cnt = 0
    images_names = sorted(glob.glob('/home/ml/dds/*_sat.jpg'))
    gt_name = sorted(glob.glob('/home/ml/dds/*_mask.png'))
    tf = A.CLAHE(p=0)
    for _ in range(5000):
        for im_n, gt_n in zip(images_names, gt_name):
            pil_im = Image.open(im_n)
            pil_gt = Image.open(gt_n).convert('L')
            image =  transforms.ToTensor()(pil_im)
            gt = torch.tensor(np.where(np.array(pil_gt) >=  1, 1., 0.)).unsqueeze(0)
            transform = transforms.RandomResizedCrop(400, scale=(0.1, 0.3), ratio=(1.,1.))
            state = torch.get_rng_state()
            image = transform(image)
            torch.set_rng_state(state)
            gt = transform(gt)
            if gt.mean() < 0.05: continue
            gt = np.array(gt*255, dtype=np.uint8)[0]

            im = Image.fromarray(np.array(gt, dtype=np.uint8))
            im.save(f'training/training/groundtruth/custom-{cnt}-mask.png')
            npim = np.moveaxis(np.array(image*255, dtype=np.uint8), 0, -1)
            npim = Image.fromarray(np.array(tf(image=np.array(npim, dtype=np.uint8))['image'], dtype=np.uint8))
            npim.save(f'training/training/images/custom-{cnt}-input.png')
            cnt += 1
            print(f'{cnt}/{10000}')
            if cnt > 10000:
                exit()