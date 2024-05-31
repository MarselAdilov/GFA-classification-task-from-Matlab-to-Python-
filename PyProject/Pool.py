import numpy as np
import scipy

def Pool(x):
    xrow, xcol, numFilters = x.shape
    y = np.zeros((int(xrow / 2), int(xcol / 2), numFilters))
    for k in range(numFilters):
        filter_n = np.ones((2, 2)) / (2 * 2)
        image = scipy.signal.convolve2d(x[:, :, k], filter_n, 'valid')
        y[:, :, k] = image[::2, ::2]
    return y
