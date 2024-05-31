import numpy as np
import scipy

def Conv(x, W):
    wrow, wcol, numFilters = W.shape
    xrow, xcol = x.shape
    yrow = xrow - wrow + 1
    ycol = xcol - wcol + 1
    y = np.zeros((yrow, ycol, numFilters))
    for k in range(numFilters):
        filter = W[:, :, k]
        filter = np.rot90(np.squeeze(filter), 2)
        y[:, :, k] = scipy.signal.convolve2d(x, filter, 'valid')
    return y
