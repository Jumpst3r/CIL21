from PIL import Image, ImageOps
import numpy as np
import os
import glob
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt

PATCH_SIZE = 16

example_neg = 0
example_pos = 0

def patchwise_extract(image_path: str, gt_path: str):
    """This function extracts features (r,g,b mean and std -> 6 features) from PATCH_SIZExPATCH_SIZE patches.

    The results are saved to disk as a numpy array named {img_name}_fts.npy
    Where label is either 0 or 1 depending on the rounded mean ground truth information for that patch
    The saved array is of shape ((400/PATCH_SIZE)**2,7), 6 features + class (label) for every patch.

    Args:
        image_path (str): Path to 400x400x3 RGB image
        groud_truth_path (str): Path to 400x00 binary image containing labels {0,1} for each pixel.
    """
    global example_neg, example_pos
    basename = image_path[image_path.find('sat'):-4]
    im = Image.open(image_path)
    im = ImageOps.grayscale(im)
    np_im = np.array(im) # array of shape (400,400,3)
    opencvImage = np_im.copy()
    gt = Image.open(gt_path)
    np_gt = np.array(gt) # array of shape (400,400,1)
    np_gt = np.where(np_gt >=  1, 1, 0)
    stride = PATCH_SIZE // 2
    assert 400 % stride == 0, 'image size not divisible by 400'
    features = np.zeros(shape=(int(400//stride)**2, 2))
    labels = np.zeros(shape=(int(400//stride)**2))
    ft_idx = 0
    for y in range(0,400-20,stride):
        for x in range(0,400-20,stride):
                patch = np_im[y:y+PATCH_SIZE,x:x+PATCH_SIZE]
                labels[ft_idx] = 1 if np.mean(np_gt[y:y+PATCH_SIZE,x:x+PATCH_SIZE]) >= 0.3 else 0 
                if labels[ft_idx] == 1:
                    example_pos += 1
                else:
                     example_neg += 1
                print('=======')
                print(labels[ft_idx])
                cv2.rectangle(opencvImage, (x, y), (x+PATCH_SIZE, y+PATCH_SIZE), (255,255,255, 0.4) if labels[ft_idx] == 1 else (0,0,0,0.4))
                features[ft_idx] = np.array([patch.mean(), patch.std()])
                ft_idx += 1
                print(np.mean(np_gt[y:y+PATCH_SIZE,x:x+PATCH_SIZE]))
                print("======")
    #cv2.imshow('', opencvImage)
    #cv2.waitKey()
    np.save('patches' + os.path.sep + (basename + '_fts'), features)
    np.save('patches' + os.path.sep + (basename + '_labels'), labels)
    
def generate_training_data(training_folder: str, gt_folder: str):
    """Calls patchwise_extract() on every image/gt pair found in the two directories passed as args

    Args:
        training_folder (str): Path to folder containing training images
        gt_folder (str): Path to folder containing ground-truth
    """    
    input_files = sorted(glob.glob(training_folder + '*.png'))
    gt_files = sorted(glob.glob(gt_folder + '*.png'))

    for (image_path, gt_path) in tqdm(zip(input_files, gt_files), total=len(input_files)):
        patchwise_extract(image_path, gt_path)

if __name__ == '__main__':
    gt_folder = 'training/training/groundtruth/'
    im_folder = 'training/training/images/'
    generate_training_data(im_folder, gt_folder)
    print("road patches: ",example_pos, "non-road patches:", example_neg)