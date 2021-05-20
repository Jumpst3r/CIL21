"""
SOURCE: https://github.com/kunalmessi10/FCN-with-CRF-post-processing/blob/master/crf.py
"""

import numpy as np
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils

MAX_ITER = 10


def dense_crf(img, sigm):
    h = sigm.shape[0]
    w = sigm.shape[1]

    p = np.zeros((2, h, w)).astype(np.float32)
    p[0] = sigm
    p[1] = 1 - sigm

    d = dcrf.DenseCRF2D(w, h, 2)

    # U = -np.log(p)
    U = utils.unary_from_softmax(p.reshape(2, -1))
    # U = U.reshape((2,-1))
    d.setUnaryEnergy(U)

    img = np.ascontiguousarray(img)

    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=3, srgb=3, rgbim=img, compat=3)

    Q = d.inference(MAX_ITER)
    Q = np.array(Q).reshape((2, h, w))

    map = np.array(Q[0] > Q[1], dtype=int)

    # plt.imshow(map)

    return map
