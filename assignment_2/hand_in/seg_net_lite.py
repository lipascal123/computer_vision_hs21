from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn
from collections import OrderedDict

logger = logging.getLogger(__name__)


class SegNetLite(nn.Module):

    def __init__(self, kernel_sizes=[3, 3, 3, 3], down_filter_sizes=[32, 64, 128, 256],
            up_filter_sizes=[128, 64, 32, 32], conv_paddings=[1, 1, 1, 1],
            pooling_kernel_sizes=[2, 2, 2, 2], pooling_strides=[2, 2, 2, 2], **kwargs):
        """Initialize SegNet Module

        Args:
            kernel_sizes (list of ints): kernel sizes for each convolutional layer in downsample/upsample path.
            down_filter_sizes (list of ints): number of filters (out channels) of each convolutional layer in the downsample path.
            up_filter_sizes (list of ints): number of filters (out channels) of each convolutional layer in the upsample path.
            conv_paddings (list of ints): paddings for each convolutional layer in downsample/upsample path.
            pooling_kernel_sizes (list of ints): kernel sizes for each max-pooling layer and its max-unpooling layer.
            pooling_strides (list of ints): strides for each max-pooling layer and its max-unpooling layer.
        """
        super(SegNetLite, self).__init__()
        self.num_down_layers = len(kernel_sizes)
        self.num_up_layers = len(kernel_sizes)

        input_size = 3 # initial number of input channels
        # Construct downsampling layers.
        # As mentioned in the assignment, blocks of the downsampling path should have the
        # following output dimension (igoring batch dimension):
        # 3 x 64 x 64 (input) -> 32 x 32 x 32 -> 64 x 16 x 16 -> 128 x 8 x 8 -> 256 x 4 x 4
        # each block should consist of: Conv2d->BatchNorm2d->ReLU->MaxPool2d
        layers_conv_down = [nn.Conv2d(3,32,3,1,1), 
                            nn.Conv2d(32,64,3,1,1), 
                            nn.Conv2d(64,128,3,1,1), 
                            nn.Conv2d(128,256,3,1,1)]
        layers_bn_down = [nn.BatchNorm2d(32), nn.BatchNorm2d(64),
                          nn.BatchNorm2d(128), nn.BatchNorm2d(256)]
        layers_pooling = [nn.MaxPool2d(2, return_indices = True),
                          nn.MaxPool2d(2, return_indices=True),
                          nn.MaxPool2d(2, return_indices=True),
                          nn.MaxPool2d(2, return_indices=True)]

        # raise NotImplementedError('Downsampling layers are not implemented!')

        # Convert Python list to nn.ModuleList, so that PyTorch's autograd
        # package can track gradients and update parameters of these layers
        self.layers_conv_down = nn.ModuleList(layers_conv_down)
        self.layers_bn_down = nn.ModuleList(layers_bn_down)
        self.layers_pooling = nn.ModuleList(layers_pooling)

        # Construct upsampling layers
        # As mentioned in the assignment, blocks of the upsampling path should have the
        # following output dimension (igoring batch dimension):
        # 256 x 4 x 4 (input) -> 128 x 8 x 8 -> 64 x 16 x 16 -> 32 x 32 x 32 -> 32 x 64 x 64
        # each block should consist of: MaxUnpool2d->Conv2d->BatchNorm2d->ReLU
        layers_conv_up = [nn.Conv2d(256,128,3,1,1), 
                            nn.Conv2d(128,64,3,1,1), 
                            nn.Conv2d(64,32,3,1,1),
                            nn.Conv2d(32,32,3,1,1)]
        layers_bn_up = [nn.BatchNorm2d(128), nn.BatchNorm2d(64),
                        nn.BatchNorm2d(32), nn.BatchNorm2d(32)]
        layers_unpooling = [nn.MaxUnpool2d(2),
                            nn.MaxUnpool2d(2),
                            nn.MaxUnpool2d(2),
                            nn.MaxUnpool2d(2)]

        # raise NotImplementedError('Upsampling layers are not implemented!')

        # Convert Python list to nn.ModuleList, so that PyTorch's autograd
        # can track gradients and update parameters of these layers
        self.layers_conv_up = nn.ModuleList(layers_conv_up)
        self.layers_bn_up = nn.ModuleList(layers_bn_up)
        self.layers_unpooling = nn.ModuleList(layers_unpooling)

        self.relu = nn.ReLU(True)

        # Implement a final 1x1 convolution to to get the logits of 11 classes (background + 10 digits)
        self.layer_final_conv = nn.Conv2d(32,11,1,1,0)

        # raise NotImplementedError('Final convolution layer is not implemented!')

    def forward(self, x):
        # Encoder
        x = self.layers_conv_down[0](x)
        x = self.layers_bn_down[0](x)
        x = self.relu(x)
        x, indices1 = self.layers_pooling[0](x)

        x = self.layers_conv_down[1](x)
        x = self.layers_bn_down[1](x)
        x = self.relu(x)
        x, indices2 = self.layers_pooling[1](x)

        x = self.layers_conv_down[2](x)
        x = self.layers_bn_down[2](x)
        x = self.relu(x)
        x, indices3 = self.layers_pooling[2](x)

        x = self.layers_conv_down[3](x)
        x = self.layers_bn_down[3](x)
        x = self.relu(x)
        x, indices4 = self.layers_pooling[3](x)

        # Decoder
        x = self.layers_unpooling[0](x,indices4)
        x = self.layers_conv_up[0](x)
        x = self.layers_bn_up[0](x)
        x = self.relu(x)

        x = self.layers_unpooling[1](x,indices3)
        x = self.layers_conv_up[1](x)
        x = self.layers_bn_up[1](x)
        x = self.relu(x)

        x = self.layers_unpooling[2](x, indices2)
        x = self.layers_conv_up[2](x)
        x = self.layers_bn_up[2](x)
        x = self.relu(x)

        x = self.layers_unpooling[3](x, indices1)
        x = self.layers_conv_up[3](x)
        x = self.layers_bn_up[3](x)
        x = self.relu(x)

        # Final Conv Layer
        x = self.layer_final_conv(x)

        return x
        # raise NotImplementedError('Forward function not implemented!')


def get_seg_net(**kwargs):

    model = SegNetLite(**kwargs)

    return model
