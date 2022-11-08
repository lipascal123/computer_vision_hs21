import torch
import torch.nn as nn
import math

class Vgg(nn.Module):
    def __init__(self, fc_layer=512, classes=10):
        super(Vgg, self).__init__()
        """ Initialize VGG simplified Module
        Args: 
            fc_layer: input feature number for the last fully MLP block
            classes: number of image classes
        """
        self.fc_layer = fc_layer
        self.classes = classes

        # todo: construct the simplified VGG network blocks
        # input shape: [bs, 3, 32, 32]
        # layers and output feature shape for each block:
        # # conv_block1 (Conv2d, ReLU, MaxPool2d) --> [bs, 64, 16, 16]
        # # conv_block2 (Conv2d, ReLU, MaxPool2d) --> [bs, 128, 8, 8]
        # # conv_block3 (Conv2d, ReLU, MaxPool2d) --> [bs, 256, 4, 4]
        # # conv_block4 (Conv2d, ReLU, MaxPool2d) --> [bs, 512, 2, 2]
        # # conv_block5 (Conv2d, ReLU, MaxPool2d) --> [bs, 512, 1, 1]
        # # classifier (Linear, ReLU, Dropout2d, Linear) --> [bs, 10] (final output)

        # hint: stack layers in each block with nn.Sequential, e.x.:
        # # self.conv_block1 = nn.Sequential(
        # #     layer1,
        # #     layer2,
        # #     layer3,
        # #     ...)

        self.conv_block1 = nn.Sequential(
                                    nn.Conv2d(3,64,3,1,1),
                                    nn.ReLU(True),
                                    nn.MaxPool2d(2) )

        self.conv_block2 = nn.Sequential(
                                    nn.Conv2d(64,128,3,1,1),
                                    nn.ReLU(True),
                                    nn.MaxPool2d(2))

        self.conv_block3 = nn.Sequential(
                                    nn.Conv2d(128,256,3,1,1),
                                    nn.ReLU(True),
                                    nn.MaxPool2d(2))

        self.conv_block4 = nn.Sequential(
                                    nn.Conv2d(256,512,3,1,1),
                                    nn.ReLU(True),
                                    nn.MaxPool2d(2))

        self.conv_block5 = nn.Sequential(
                                    nn.Conv2d(512,512,3,1,1),
                                    nn.ReLU(True),
                                    nn.MaxPool2d(2))

        self.classifier = nn.Sequential(
                                    nn.Linear(512,100), # output size can be chosen
                                    nn.ReLU(True),
                                    nn.Dropout(0.2), # drop out probability can be chosen
                                    nn.Linear(100,10))
       

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
        """
        :param x: input image batch tensor, [bs, 3, 32, 32]
        :return: score: predicted score for each class (10 classes in total), [bs, 10]
        """
        score = None
        # todo
        # print("0:",x.shape)
        x = self.conv_block1(x)
        # print("1:",x.shape)
        x = self.conv_block2(x)
        # print("2:",x.shape)
        x = self.conv_block3(x)
        # print("3:",x.shape)
        x = self.conv_block4(x)
        # print("4:",x.shape)
        x = self.conv_block5(x)
        # print("5:",x.shape)
        # Note: reshape from [bs,512,1,1] to [bs,512] needed
        score = self.classifier(x.view(x.shape[0],-1))
        # print("6:",score.shape)
        
        return score

