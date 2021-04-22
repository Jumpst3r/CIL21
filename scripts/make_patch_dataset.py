import os
import cv2
import imageio
import numpy as np
import os.path as osp

input_dir = r'../training'
output_dir = r'../training_patch'
for phase in os.listdir(input_dir):
    for set in os.listdir(osp.join(input_dir, phase)):
        if set == 'images':
            interp = cv2.INTER_LINEAR
        elif set == 'groundtruth':
            interp = cv2.INTER_NEAREST
            
        in_folder = osp.join(input_dir, phase, set)
        out_folder = osp.join(output_dir, phase, set)

        for img_name in os.listdir(in_folder):
            img = imageio.imread(osp.join(in_folder, img_name))
            img = cv2.resize(img, (608,608), interpolation=interp)
            if set == 'images':
                os.makedirs(out_folder, exist_ok=True)
                imageio.imwrite(osp.join(out_folder, img_name), img)
            elif set == 'groundtruth':
                os.makedirs(out_folder.replace('groundtruth', 'groundtruth_full'), exist_ok=True)
                imageio.imwrite(osp.join(out_folder.replace('groundtruth', 'groundtruth_full'), img_name), img)

                # these need to be patched manually by mask_to_submission.py & submission_to_mask.py

input_dir = r'../training_patch/training/groundtruth'
output_dir = r'../training_patch/training/groundtruth_lr'
os.makedirs(output_dir, exist_ok=True)
for img_file in os.listdir(input_dir):
    img_path = osp.join(input_dir, img_file)
    img_path_out = osp.join(output_dir, img_file)
    img = imageio.imread(img_path)
    img = cv2.resize(img, (38,38), interpolation=cv2.INTER_NEAREST)
    imageio.imwrite(img_path_out, img)
