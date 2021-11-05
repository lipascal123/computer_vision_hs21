import time
import os
import random
import math
import torch
import numpy as np

# run `pip install scikit-image` to install skimage, if you haven't done so
from skimage import io, color
from skimage.transform import rescale

def distance(x, X):
    """
    Compute the distance between given point x and all other points in X

    Arguments:
    x: torch tensor representing a specific point, 1x3
    X: a torch tensor representing the image / all other points, nx3
    
    Return:
    dist: distance as numpy array nx1
    """

    dist = np.linalg.norm((X-x),axis=1, ord=2) 

    assert(X.shape[0] == dist.shape[0])

    return dist


def distance_batch(x, X):
    raise NotImplementedError('distance_batch function not implemented!')

def gaussian(dist, bandwidth):
    """
    Computes weights for points depending on their distance with a gaussian kernel

    Arguments:
    dist: The distance (2-norm) nx1 array
    bandwidth: The bandwith of the guassian kernel

    Return:
    K: Weighted particles as numpy array nx1

    """
    K = (1/(np.sqrt(2*np.pi)*bandwidth))*np.exp( (-0.5*np.power(dist,2)) / np.power(bandwidth,2) )
    
    return K

def update_point(weight, X):
    """
    Computes the weighted mean

    Return:
    
    m: updated mean as torch tensor 1x3
    """
    
    nomi = np.dot(weight.T, X)
    denomi = np.sum(weight)
    m = torch.tensor([nomi/denomi])

    return m


def update_point_batch(weight, X):
    raise NotImplementedError('update_point_batch function not implemented!')

def meanshift_step(X, bandwidth=2.5):
    X_ = X.clone()
    for i, x in enumerate(X):
        dist = distance(x, X)
        weight = gaussian(dist, bandwidth)
        X_[i] = update_point(weight, X)
    return X_

def meanshift_step_batch(X, bandwidth=2.5):
    rng = np.random.default_rng()
    idx = rng.choice(X.shape[0]-1, size=1500, replace=False)
    X_b = X[idx]
    X_ = X.clone()
    for i,x in enumerate(X_b):
        dist = distance(x, X_b)
        weight = gaussian(dist, bandwidth)
        X_[idx[i]] = update_point(weight, X_b)

    return X_
    

def meanshift(X):
    X = X.clone()
    for _ in range(20):
        # X = meanshift_step(X)   # slow implementation
        X = meanshift_step_batch(X)   # fast implementation
    X = meanshift_step(X)    
    return X

scale = 0.25    # downscale the image to run faster

# Load image and convert it to CIELAB space
image = rescale(io.imread('cow.jpg'), scale, multichannel=True)
image_lab = color.rgb2lab(image)
shape = image_lab.shape # record image shape
image_lab = image_lab.reshape([-1, 3])  # flatten the image


# Run your mean-shift algorithm
t = time.time()
X = meanshift(torch.from_numpy(image_lab)).detach().cpu().numpy()
# X = meanshift(torch.from_numpy(data).cuda()).detach().cpu().numpy()  # you can use GPU if you have one
t = time.time() - t
print ('Elapsed time for mean-shift: {}'.format(t))

# Load label colors and draw labels as an image
colors = np.load('colors.npz')['colors']
colors[colors > 1.0] = 1
colors[colors < 0.0] = 0

centroids, labels = np.unique((X / 4).round(), return_inverse=True, axis=0)

print("labels shape: ", labels.shape)
print("shape of shape: ", shape)
print("shape centroids: ", centroids.shape)
print("colors shape: ", colors.shape)
print("max of labels:", np.amax(labels))
result_image = colors[labels].reshape(shape)
result_image = rescale(result_image, 1 / scale, order=0, multichannel=True)     # resize result image to original resolution
result_image = (result_image * 255).astype(np.uint8)
io.imsave('result.png', result_image)
