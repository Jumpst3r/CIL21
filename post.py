import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import (unary_from_softmax, unary_from_labels)
from skimage.color import gray2rgb, rgba2rgb
from skimage import img_as_ubyte
import numpy.random as rd
NB_ITERATIONS = 50
"""
Function which returns the labelled image after applying CRF
adapted from https://github.com/lucasb-eyer/pydensecrf/tree/master/pydensecrf
"""


def crf(original_image, annotated_image):
    original_image = np.moveaxis(original_image.detach().cpu().numpy(), 0, -1)
    original_image = np.array(original_image, dtype=np.uint8)
    # annotated_image = np.moveaxis(annotated_image, -1, 0)
    probas = annotated_image


    d = dcrf.DenseCRF2D(original_image.shape[1], original_image.shape[0], 2)

    U = unary_from_softmax(probas.reshape(2, -1))
    d.setUnaryEnergy(U)

    d.addPairwiseBilateral(sxy=1, srgb=100, rgbim=original_image, compat=5)

    # This adds the color-independent term, features are the locations only.
    d.addPairwiseGaussian(sxy=2, compat=2)

    # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
    # d.addPairwiseBilateral(sxy=5, srgb=50, rgbim=original_image, compat=10)

    Q = d.inference(NB_ITERATIONS)

    MAP = np.array(Q).reshape(2, original_image.shape[0], original_image.shape[1])

    result = np.zeros((2, original_image.shape[0], original_image.shape[1]))

    result = np.array(MAP[0] < MAP[1], dtype=int)

    return result