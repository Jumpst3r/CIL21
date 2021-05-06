import os
import os.path as osp
import imageio
import glob
import shutil
import random


def get_data_paths(img_dir, labels_dir):
    img_paths = sorted(glob.glob(img_dir + '*.png'))
    label_paths = sorted(glob.glob(labels_dir + '*.png'))
    data_paths = list(zip(img_paths, label_paths))
    # random.shuffle(data_paths) # I removed the shuffling before the split, so I can more accurately compare models
    return data_paths


full = r'../training/full'
train = r'../training/train'
val = r'../training/val'

os.makedirs(train + '/images')
os.makedirs(train + '/groundtruth')
os.makedirs(val + '/images')
os.makedirs(val + '/groundtruth')

full_dp = get_data_paths(full + '/images/', full + '/groundtruth/')

train_idx = sorted(random.sample(range(0, 100), 70))

for i, (img_path, label_path) in enumerate(full_dp):
    if i in train_idx:
        shutil.copy(img_path, img_path.replace('full', 'train'))
        shutil.copy(label_path, label_path.replace('full', 'train'))
    else:
        shutil.copy(img_path, img_path.replace('full', 'val'))
        shutil.copy(label_path, label_path.replace('full', 'val'))

