# Computer Vision Programming Assignments HS 2021

## Assignment 1

The goal of this assignemnt was to get familiar with pytorch.

### 2D Classifier
In this exercise I trained a classifier for a 2D dataset. The data is not linearly separable. Therefore a multi layer perceptron classifier with 2 hidden layers is trained.

<img src="https://user-images.githubusercontent.com/43472532/140576104-52ef0989-fa5c-4a67-b7ff-f4807aecce24.jpg" width="250">

### Digit Classifier
in the second exercise I trained a convolutional neural network to classifiy the hand written digits of the [MNIST dataset](https://en.wikipedia.org/wiki/MNIST_database).

<img src="https://user-images.githubusercontent.com/43472532/140576626-87133491-9596-4f63-b20a-b1624ccb9d91.png" width="350">


## Assignment 2

This Assignment is about image segmentation

### Mean Shift 
In this exercise the mean shift algorithm is implemented to cluster pixels. A batchified version running on a GPU with pytorch and cuda achieves a runtime of 3 seconds to segment a cow image.

### SegNet
A lite version of the SegNet is used to perform image segmenteation on the MNIST dataset. An accuracy of 86% (IoU) is achieved.

## Assignment 3
In this exercise I calibrated a camera and a structre from motion (sfm) problem where I reconstruced a small scene.

### Structure from Motion
The left picture shows one of the input images and the other picture the reconstructed scene.

ADD PICTURES


## Assignment 4
In the first part I implemented a RANSAC for robust model fitting. In the second part I solved a multi-view stereo problem using deep learning.

ADD TWO PICTURES

## Assignment 5
The first part is the implementation of a bag-of-words classifier to test if an image contains a car or not. The second part is a CNN-based image classification on CIFAR-10 dataset.

## Assignment 6
In this exercise we solved a tracking problem using sampled-based solution of the recursive Bayesian filter. This allowed us to e.g. track the hand in the image depicted below.

ADD PICTURE


