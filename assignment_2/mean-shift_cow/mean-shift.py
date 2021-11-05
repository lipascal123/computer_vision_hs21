import time
import os
import random
import math
import torch
import numpy as np

# run `pip install scikit-image` to install skimage, if you haven't done so
from skimage import io, color
from skimage.transform import rescale

def distance(x, X, device):
    """
    Compute the distance between given point x and all other points in X

    Arguments:
    x: torch tensor representing a specific point, 1x3
    X: a torch tensor representing the image / all other points, nx3
    
    Return:
    dist: distance as torch tensor nx1
    """

    # numpy norm is faster on cpu than torch norm
    if device == "cpu":
        dist = np.linalg.norm((X-x),axis=1, ord=2)  
        return torch.tensor(dist)

    else:
        dist = torch.linalg.norm((X-x), dim=1, ord=None, keepdim=True)
        return dist
    
    
def gaussian(dist, bandwidth):
    """
    Computes weights for points depending on their distance with a gaussian kernel

    Arguments:
    dist: The distance (2-norm) nx1 torch tensor
    bandwidth: The bandwith of the guassian kernel

    Return:
    K: Weighted particles as torch tensor nx1

    """
    K = torch.exp( (-0.5*torch.pow(dist,2)) / (bandwidth**2) )

    return K

def update_point(weight, X):
    """
    Computes the weighted mean

    Return:
    
    m: updated mean as torch tensor 1x3
    """
 
    nomi = torch.matmul(weight.T,X)   
    denomi = torch.sum(weight)

    return nomi/denomi


def update_point_batch(weight, X):
    """
    Computes the weighted mean for each pixel in the batch

    Return:
    
    m: updated mean as torch tensor nx3
    """
    
    weight_new = weight.view(-1,X.shape[0])
    nomi = torch.matmul(weight_new,X)
    denomi = torch.sum(weight_new, dim=1, keepdim=True)

    return nomi/denomi

def meanshift_step(X, device, bandwidth=2.5):
    """
    Performs one step of the meanshift in a for-loop fashion.

    Return:

    X_: updated torch tensor with new weighted means 
    """
    X_ = X.clone()
    for i, x in enumerate(X):
        dist = distance(x, X, device)
        weight = gaussian(dist, bandwidth)
        X_[i] = update_point(weight, X)
    return X_

def meanshift_step_batch(X, device, bandwidth=2.5):
    """
    Performs one step of the meanshift in batch fashion.

    Return:

    X_: updated torch tensor with new weighted means 
    """
   
    # Build pytorch tensor
    x_batch = X.repeat_interleave(X.shape[0], dim = 0) # n*n x 3
    X_batch = X.repeat(X.shape[0],1) # n*n x 3

    # compute mean shift 
    dist = distance(X_batch, x_batch, device) # n*n x 1
    weight = gaussian(dist, bandwidth) # n*n x 1
    X_ = update_point_batch(weight, X) # n x 3
    
    return X_
    

def meanshift(X, device):
    X = X.clone()
    # print(X.shape)
    for _ in range(20):
        # X = meanshift_step(X, device)   # slow implementation
        X = meanshift_step_batch(X, device)   # fast implementation   
    return X

scale = 0.25    # downscale the image to run faster

# Check CUDA
device = "cpu"
if torch.cuda.is_available():
    device = "cuda" 
print("Device:", device)

# Load image and convert it to CIELAB space
image = rescale(io.imread('cow.jpg'), scale, multichannel=True)
image_lab = color.rgb2lab(image)
shape = image_lab.shape # record image shape
image_lab = image_lab.reshape([-1, 3])  # flatten the image


# Run your mean-shift algorithm
t = time.time()
# X = meanshift(torch.from_numpy(image_lab)).detach().cpu().numpy()
X = meanshift(torch.from_numpy(image_lab).to(device),device).detach().cpu().numpy()  # you can use GPU if you have one
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
