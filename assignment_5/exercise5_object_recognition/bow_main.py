import numpy as np
import cv2
import glob
import os
from sklearn.cluster import KMeans
from tqdm import tqdm
from scipy import spatial # added additionally
import matplotlib.pyplot as plt # added additionally


def findnn(D1, D2):
    """
    :param D1: NxD matrix containing N feature vectors of dim. D
    :param D2: MxD matrix containing M feature vectors of dim. D
    :return:
        Idx: N-dim. vector containing for each feature vector in D1 the index of the closest feature vector in D2.
        Dist: N-dim. vector containing for each feature vector in D1 the distance to the closest feature vector in D2
    """
    N = D1.shape[0]
    M = D2.shape[0]  # [k]

    # Find for each feature vector in D1 the nearest neighbor in D2
    Idx, Dist = [], []
    for i in range(N):
        minidx = 0
        mindist = np.linalg.norm(D1[i, :] - D2[0, :])
        for j in range(1, M):
            d = np.linalg.norm(D1[i, :] - D2[j, :])

            if d < mindist:
                mindist = d
                minidx = j
        Idx.append(minidx)
        Dist.append(mindist)
    return Idx, Dist


def grid_points(img, nPointsX, nPointsY, border):
    """
    :param img: input gray img, numpy array, [h, w]
    :param nPointsX: number of grids in x dimension
    :param nPointsY: number of grids in y dimension
    :param border: leave border pixels in each image dimension
    :return: vPoints: 2D grid point coordinates, numpy array, [nPointsX*nPointsY, 2]
    """
    vPoints = None  # numpy array, [nPointsX*nPointsY, 2]

    # todo

    # shape of image
    w = img.shape[1] 
    h = img.shape[0]

    x_grid_points = []
    y_grid_points = []

    # interval size in x and y direction
    x_interval = (w-2*border)/(nPointsX-1)
    y_interval = (h-2*border)/(nPointsY-1)
 
    # compute x grid point coordinates -> width
    for i in range(nPointsX): 
        x_grid_points.append( round(border+x_interval*i) )

    # compute y grid point coordinates -> height
    for j in range(nPointsY):
        y_grid_points.append( round(border+y_interval*j) )
        
    # stack y and x grid point coordinates
    x_grid, y_grid = np.meshgrid(x_grid_points,y_grid_points)
    # plt.plot(x_grid,y_grid, marker='.', color='k', linestyle='none')
    # plt.title(("height=y", h,"width=x", w))
    # plt.show()
    x_grid_vec = x_grid.reshape(-1)
    y_grid_vec = y_grid.reshape(-1)
    vPoints = np.stack((y_grid_vec, x_grid_vec), axis=1)
    return vPoints # [nPointsX*nPointsY, 2] height,width


def descriptors_hog(img, vPoints, cellWidth, cellHeight):
    """
    :param img: input gray img, numpy array, [h, w]
    :param vPoints: 2D grid point coordinates, numpy array, [nPointsX*nPointsY, 2]
    :param cellWidth: width of one cell for gradient computation
    :param cellHeight: height of one cell for gradient computation
    :return descriptors: descriptor for all grid points in img [nPointsX*nPointsY, 128]
    """
    nBins = 8
    w = cellWidth
    h = cellHeight

    # grad_x = cv2.Sobel(img, cv2.CV_16S, dx=1, dy=0, ksize=1) # numpy nd array [h,w]
    # grad_y = cv2.Sobel(img, cv2.CV_16S, dx=0, dy=1, ksize=1) # numpy nd array [h,w]

    # todo
    # I changed the type to CV_64F s.t. I don't get an overflow when grad_x is squared 
    grad_x = cv2.Sobel(img, cv2.CV_64F, dx=1, dy=0, ksize=1) # numpy nd array [h,w]
    grad_y = cv2.Sobel(img, cv2.CV_64F, dx=0, dy=1, ksize=1) # numpy nd array [h,w]

    # compute magnitude
    mag = np.sqrt(grad_x**2 + grad_y**2) # [h,w]
    orient = np.arctan2(grad_y, grad_x) # [h,w]

    descriptors = []  # list of descriptors for the current image, each entry is one 128-d vector for a grid point
    for i in range(len(vPoints)):
        # image coordinates of grid point
        y,x = vPoints[i,:] # []
        # magnitude and orientation of pixels in surrounding cells of grid point
        mag_cells = mag[y-2*h:y+2*h,x-2*w:x+2*w] # [4*h,4*w]
        orient_cells = orient[y-2*h:y+2*h,x-2*w:x+2*w] # [4*h,4*w]
    
        hog_grid_point = []
        for i in [0,h,2*h,3*h]: # row-axis
            for j in [0,w,2*w,3*w]: # col-axis
                mag_cell = mag_cells[i:i+h,j:j+w]
                orient_cell = orient_cells[i:i+h,j:j+w]
                hist_cell, _ = np.histogram(orient_cell, bins=nBins, weights=mag_cell, density=False)
                # normalize 
                if hist_cell.sum() != 0:
                    hist_cell = hist_cell / hist_cell.sum()
                    # print("hist_cell",hist_cell)
                    # print("mag_cell",mag_cell)
                    # print("mag_cells",mag_cells)
                    # print("orient_cell",orient_cell)
                    # assert(False)
                hog_grid_point.append(hist_cell)
        
        descriptors.append(hog_grid_point)

    descriptors = np.asarray(descriptors) # descriptor for the current image (100 grid points)
    # reshape to [nPointsX*nPointsY, 128]
    descriptors = descriptors.reshape(vPoints.shape[0],descriptors.shape[1]*descriptors.shape[2])
    return descriptors # [nPointsX*nPointsY, 128]


def create_codebook(nameDirPos, nameDirNeg, k, numiter):
    """
    :param nameDirPos: dir to positive training images
    :param nameDirNeg: dir to negative training images
    :param k: number of kmeans cluster centers
    :param numiter: maximum iteration numbers for kmeans clustering
    :return: vCenters: center of kmeans clusters, numpy array, [k, 128]
    """
    vImgNames = sorted(glob.glob(os.path.join(nameDirPos, '*.png')))
    vImgNames = vImgNames + sorted(glob.glob(os.path.join(nameDirNeg, '*.png')))

    nImgs = len(vImgNames)

    cellWidth = 4
    cellHeight = 4
    nPointsX = 10
    nPointsY = 10
    border = 8

    vFeatures = []  # list for all features of all images (each feature: 128-d, 16 histograms containing 8 bins)
    # Extract features for all image
    for i in tqdm(range(nImgs)):
        # print('processing image {} ...'.format(i+1))
        img = cv2.imread(vImgNames[i])  # [172, 208, 3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # [h, w]

        # Collect local feature points for each image, and compute a descriptor for each local feature point
        # todo
        # get the img coordinates of the grid points
        vPoints = grid_points(img, nPointsX, nPointsY, border) # [nPointsX*nPointsY, 2]
        # compute descriptors for each grid points
        descriptor = descriptors_hog(img, vPoints, cellWidth, cellHeight) # [nPointsX*nPointsY, 128]
        vFeatures.append(descriptor)

    vFeatures = np.asarray(vFeatures)  # [n_imgs, n_vPoints, 128]
    vFeatures = vFeatures.reshape(-1, vFeatures.shape[-1])  # [n_imgs*n_vPoints, 128]
    print('number of extracted features: ', len(vFeatures))


    # Cluster the features using K-Means
    print('clustering ...')
    kmeans_res = KMeans(n_clusters=k, max_iter=numiter).fit(vFeatures)
    vCenters = kmeans_res.cluster_centers_  # [k, 128]
    return vCenters


def bow_histogram(vFeatures, vCenters):
    """
    :param vFeatures: MxD matrix containing M feature vectors of dim. D
    :param vCenters: NxD matrix containing N cluster centers of dim. D
    :return: histo: N-dim. numpy vector containing the resulting BoW activation histogram.
    """
    histo = np.zeros(vCenters.shape[0])

    # todo
    # use KDtree to speed up similarity comparsion of vectors
    tree = spatial.KDTree(vCenters)
    for i in range(vFeatures.shape[0]):
        closest_vw = tree.query(vFeatures[i,:])[1]
        histo[closest_vw] += 1

    return histo


def create_bow_histograms(nameDir, vCenters):
    """
    :param nameDir: dir of input images
    :param vCenters: kmeans cluster centers, [k, 128] (k is the number of cluster centers)
    :return: vBoW: matrix, [n_imgs, k]
    """
    vImgNames = sorted(glob.glob(os.path.join(nameDir, '*.png')))
    nImgs = len(vImgNames)

    cellWidth = 4
    cellHeight = 4
    nPointsX = 10
    nPointsY = 10
    border = 8

    # Extract features for all images in the given directory
    vBoW = []
    for i in tqdm(range(nImgs)):
        # print('processing image {} ...'.format(i + 1))
        img = cv2.imread(vImgNames[i])  # [172, 208, 3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # [h, w]

        # todo
        # extract hog descripor for query image
        vPoints = grid_points(img, nPointsX, nPointsY, border) # [nPointsX*nPointsY, 2]
        descriptors = descriptors_hog(img, vPoints, cellWidth, cellHeight) # [nPointsX*nPointsY, 128]
        # create bow histogram with descripors and cluster centers
        bow_hist = bow_histogram(descriptors,vCenters) # [k]
        # now the query image is described with a bag of visual words
        vBoW.append(bow_hist)


    vBoW = np.asarray(vBoW)  # [n_imgs, k]
    return vBoW



def bow_recognition_nearest(histogram,vBoWPos,vBoWNeg):
    """
    :param histogram: bag-of-words histogram of a test image, [1, k]
    :param vBoWPos: bag-of-words histograms of positive training images, [n_imgs, k]
    :param vBoWNeg: bag-of-words histograms of negative training images, [n_imgs, k]
    :return: sLabel: predicted result of the test image, 0(without car)/1(with car)
    """

    DistPos, DistNeg = None, None

    # Find the nearest neighbor in the positive and negative sets and decide based on this neighbor
    # todo
    tree_pos = spatial.KDTree(vBoWPos)
    tree_neg = spatial.KDTree(vBoWNeg)
    DistPos = tree_pos.query(histogram)
    DistNeg = tree_neg.query(histogram)

    if (DistPos < DistNeg):
        sLabel = 1
    else:
        sLabel = 0
    return sLabel





if __name__ == '__main__':
    nameDirPos_train = 'data/data_bow/cars-training-pos'
    nameDirNeg_train = 'data/data_bow/cars-training-neg'
    nameDirPos_test = 'data/data_bow/cars-testing-pos'
    nameDirNeg_test = 'data/data_bow/cars-testing-neg'


    k = 25 # todo
    numiter = 550  # todo

    print('creating codebook ...')
    vCenters = create_codebook(nameDirPos_train, nameDirNeg_train, k, numiter)

    print('creating bow histograms (pos) ...')
    vBoWPos = create_bow_histograms(nameDirPos_train, vCenters)
    print('creating bow histograms (neg) ...')
    vBoWNeg = create_bow_histograms(nameDirNeg_train, vCenters)

    # test pos samples
    print('creating bow histograms for test set (pos) ...')
    vBoWPos_test = create_bow_histograms(nameDirPos_test, vCenters)  # [n_imgs, k]
    result_pos = 0
    print('testing pos samples ...')
    for i in range(vBoWPos_test.shape[0]):
        cur_label = bow_recognition_nearest(vBoWPos_test[i:(i+1)], vBoWPos, vBoWNeg)
        result_pos = result_pos + cur_label
    acc_pos = result_pos / vBoWPos_test.shape[0]
    print('test pos sample accuracy:', acc_pos)

    # test neg samples
    print('creating bow histograms for test set (neg) ...')
    vBoWNeg_test = create_bow_histograms(nameDirNeg_test, vCenters)  # [n_imgs, k]
    result_neg = 0
    print('testing neg samples ...')
    for i in range(vBoWNeg_test.shape[0]):
        cur_label = bow_recognition_nearest(vBoWNeg_test[i:(i + 1)], vBoWPos, vBoWNeg)
        result_neg = result_neg + cur_label
    acc_neg = 1 - result_neg / vBoWNeg_test.shape[0]
    print('test neg sample accuracy:', acc_neg)
