import matplotlib.pyplot as plt
import numpy as np
import torch

def vis(sample, i=0):
    x, y = sample
    f, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')
    ax1.imshow(x[i, :3].numpy().transpose(1,2,0))
    ax2.imshow(y[i][0].numpy())
    ax3.imshow(x[i, :3].numpy().transpose(1,2,0))
    ax3.imshow(y[i][0].numpy(), alpha=0.5)
    # ax2.imshow(x[i, 3].numpy())
    # ax3.imshow(y[i][0].numpy(), cmap='binary_r')
    plt.show()


def patch_to_label(patch, foreground_threshold=0.5):
    try:
        df = torch.mean(patch)
    except:
        df = np.mean(patch)
    if df > foreground_threshold:
        return 1
    else:
        return 0


def get_accuracy(pred, gt, patch_size=16):
    assert(pred.shape == gt.shape)
    num_corr = 0
    num_patches = 0
    for j in range(0, pred.shape[1], patch_size):
        for i in range(0, pred.shape[0], patch_size):
            label_pred = patch_to_label(pred[i:i + patch_size, j:j + patch_size])
            label_gt = patch_to_label(gt[i:i + patch_size, j:j + patch_size])
            num_patches += 1
            if label_pred == label_gt:
                num_corr += 1
    return num_corr / num_patches
