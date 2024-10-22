import torch
import numpy as np

from .nntools import NNRegressor
# from .nntools import * #as nt



class DnCNN(NNRegressor):
    """ This class is an implementation of the DnCNN (Deep Convolutional neural network)

    """

    def __init__(self, D=4, C=64, image_mode=1):
        """ Initialize the DnCNN

        Arguments:
            D(int, optional)          : The number of layers
            C(int, optional)          : The number of output channel for the convolution
            image_mode(int, optional) : The number of input channel of the images
        """

        super(DnCNN, self).__init__()

        self.D=D
        self.C=C
        self.image_mode=image_mode

        self.conv = torch.nn.ModuleList()
        self.conv.append(torch.nn.Conv2d(   self.image_mode,    self.C,             3,  padding=1))
        self.conv.extend([torch.nn.Conv2d(  self.C,             self.C,             3,  padding=1) for _ in range(self.D)])
        self.conv.append(torch.nn.Conv2d(   self.C,             self.image_mode,    3,  padding=1))

        for i in range(len(self.conv[:-1])):
            torch.nn.init.kaiming_normal_(self.conv[i].weight.data, nonlinearity='relu')


        self.bn = torch.nn.ModuleList()
        self.bn.extend([torch.nn.BatchNorm2d(self.C, self.C) for _ in range(self.D)])

        for i in range(D):
            torch.nn.init.constant_(self.bn[i].weight.data, 1.25 * np.sqrt(self.C))


    def forward(self, input):
        """ Take an input and pass it through the network

        Arguments:
            input(tensor)   : The tensor that will feed the network
        Return:
            The tensor after his passage inside the network corresponding to the predicted noise
        """

        h = torch.nn.functional.relu(self.conv[0](input))
        for i in range(self.D):
            h = torch.nn.functional.relu(self.bn[i](self.conv[i+1](h)))
        return input - self.conv[self.D+1](h)
