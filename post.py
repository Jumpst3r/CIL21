import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import (unary_from_softmax)
from skimage.color import gray2rgb, rgba2rgb
from skimage import img_as_ubyte
import numpy.random as rd
NB_ITERATIONS = 20

"""
Function which returns the labelled image after applying CRF
adapted from https://github.com/lucasb-eyer/pydensecrf/tree/master/pydensecrf
"""


def crf(original_image, annotated_image):
    rd.seed(123)
    annotated_image = annotated_image.detach().numpy()
    original_image = img_as_ubyte(original_image)
    # annotated_image = np.moveaxis(annotated_image, -1, 0)
    annotated_image = annotated_image.copy(order='C')

    _tmp = np.zeros((2,annotated_image.shape[1],annotated_image.shape[2]))
    _tmp[0] = np.array(np.round(annotated_image) == 0, dtype=int)
    _tmp[1] = np.array(np.round(annotated_image) == 1, dtype=int)

    d = dcrf.DenseCRF2D(original_image.shape[1], original_image.shape[0], 2)

    

    U = unary_from_softmax(_tmp.reshape(2,-1))
    d.setUnaryEnergy(U)

    d.addPairwiseGaussian(sxy=5, compat=20, kernel=dcrf.DIAG_KERNEL,
                          normalization=dcrf.NORMALIZE_SYMMETRIC)

    d.addPairwiseBilateral(sxy=50, srgb=50, rgbim=original_image,
                           compat=15,
                           kernel=dcrf.DIAG_KERNEL,
                           normalization=dcrf.NORMALIZE_SYMMETRIC)

    Q = d.inference(NB_ITERATIONS)

    MAP = np.array(Q).reshape(2, original_image.shape[0], original_image.shape[1])

    result = np.zeros((1, original_image.shape[0], original_image.shape[1]))

    result = np.array(MAP[0] < MAP [1], dtype=int)

    return result