import matplotlib.pyplot as plt
import numpy as np


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
