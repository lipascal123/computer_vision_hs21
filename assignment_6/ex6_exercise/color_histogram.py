import numpy as np
import matplotlib.pyplot as plt


def color_histogram(xmin, ymin, xmax, ymax, frame, hist_bin):
    """
    Computes the normalized histogram for each channel in frame

    :param xmin
    :param xmax
    :param ymin
    :param ymax
    :param frame image from video stram [120,160,3]
    :param hist_bin
    """
    # crop frame to get ROI
    box = frame[ymin:ymax, xmin:xmax, :]

    hist = np.zeros((box.shape[2],hist_bin))
    # loop over channels to build histogram for each channel
    for i in range(box.shape[2]):
        # compute histogram, input frame will be flattened by np.histogram
        hist[i,:], _ = np.histogram(box[:, :, i], bins=hist_bin, range=(0, 255),density=False)
        # normalize histogram
        hist = hist / np.sum(hist,axis=1,keepdims=True)

    return hist # [3,hist_bin]
    
