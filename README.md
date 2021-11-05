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


