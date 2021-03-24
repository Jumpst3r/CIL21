import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import (unary_from_softmax, unary_from_labels)
from skimage.color import gray2rgb, rgba2rgb
from skimage import img_as_ubyte
import numpy.random as rd
NB_ITERATIONS = 2

"""
Function which returns the labelled image after applying CRF
adapted from https://github.com/lucasb-eyer/pydensecrf/tree/master/pydensecrf
"""


def crf(original_image, annotated_image):
    original_image = img_as_ubyte(original_image)
    # annotated_image = np.moveaxis(annotated_image, -1, 0)
    probas = np.zeros(shape=(2,original_image.shape[1],original_image.shape[1]), dtype=np.float32)
    probas[0] = 1 - annotated_image[0]
    probas[1] = annotated_image[0]


    d = dcrf.DenseCRF2D(original_image.shape[1], original_image.shape[0], 2)

    U = unary_from_softmax(probas.reshape(2, -1))
    d.setUnaryEnergy(U)

    # This adds the color-independent term, features are the locations only.
    d.addPairwiseGaussian(sxy=5, compat=5)

    # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
    d.addPairwiseBilateral(sxy=5, srgb=50, rgbim=original_image,
                           compat=10)

    Q = d.inference(NB_ITERATIONS)

    MAP = np.array(Q).reshape(2, original_image.shape[0], original_image.shape[1])

    result = np.zeros((2, original_image.shape[0], original_image.shape[1]))

    result = np.array(MAP[0] < MAP[1], dtype=int)

    return result