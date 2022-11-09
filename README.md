# Computer Vision Programming Assignments HS 2021

## Assignment 1

The goal of this assignemnt was to get familiar with pytorch.
<img src="https://user-images.githubusercontent.com/43472532/140576104-52ef0989-fa5c-4a67-b7ff-f4807aecce24.jpg" width="150" align="right">
### 2D Classifier

In this exercise I trained a classifier for a 2D dataset. The data is not linearly separable. Therefore a multi layer perceptron classifier with 2 hidden layers is trained.

<img src="https://user-images.githubusercontent.com/43472532/140576626-87133491-9596-4f63-b20a-b1624ccb9d91.png" width="150" align="right">

### Digit Classifier
in the second exercise I trained a convolutional neural network to classifiy the hand written digits of the [MNIST dataset](https://en.wikipedia.org/wiki/MNIST_database).



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

<p float="middle">  
  <img src="https://user-images.githubusercontent.com/43472532/200794427-5caff0ed-9081-4467-bc59-f552b6c79beb.png" width="40%" align="left">
  <img src="https://user-images.githubusercontent.com/43472532/200794721-fbb32674-721c-4dac-9664-3d1a7244c57a.png" width="25%">
</p>

## Assignment 4
In the first part I implemented a RANSAC for robust model fitting. In the second part I solved a multi-view stereo problem using deep learning.


<p float="middle">
  <img src="https://user-images.githubusercontent.com/43472532/200794104-4297cc8f-800b-4502-a0a8-e070ef3e3c5c.png" width="30%">
  <img src="https://user-images.githubusercontent.com/43472532/200793810-6e768582-1184-4462-9802-f9325cba41b6.png" width="25%">
</p>

## Assignment 5
The first part is the implementation of a bag-of-words classifier to test if an image contains a car or not. The second part is a CNN-based image classification on CIFAR-10 dataset.

## Assignment 6
In this exercise we solved a tracking problem using sampled-based solution of the recursive Bayesian filter. This allowed us to e.g. track the hand in the image depicted below.

<img src="https://user-images.githubusercontent.com/43472532/200793474-6ec85fab-72b7-476d-b2aa-5ae027855ef5.png" width="350">
