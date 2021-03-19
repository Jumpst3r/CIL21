import matplotlib.pyplot as plt
import numpy as np


def vis(sample, i=0):
    x, y = sample
    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.axis('off')
    ax2.axis('off')
    ax1.imshow(x[i].numpy().transpose(1,2,0))
    ax2.imshow(y[i][0].numpy(), cmap='binary_r')
    plt.show()
