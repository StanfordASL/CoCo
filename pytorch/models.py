import torch
import numpy as np
from    torch import nn
from    torch.nn import functional as F

class FFNet(torch.nn.Module):
    """Simple class to implement a feed-forward neural network in PyTorch.
    
    Attributes:
        layers: list of torch.nn.Linear layers to be applied in forward pass.
        activation: activation function to be applied between layers.
    
    """
    def __init__(self,shape,activation=None):
        """Constructor for FFNet.
        
        Arguments:
            shape: list of ints describing network shape, including input & output size.
            activation: a torch.nn function specifying the network activation.
        """
        super().__init__()
        self.shape = shape
        self.activation = activation ##TODO(pculbertson): make it possible use >1 activation... maybe? who cares

        self.vars = nn.ParameterList()

        for ii in range(0,len(shape)-1):
            weight_ii = nn.Parameter(torch.ones(shape[ii+1], shape[ii]))
            torch.nn.init.kaiming_normal_(weight_ii)
            self.vars.append(weight_ii)

            bias_ii = nn.Parameter(torch.zeros(shape[ii+1]))
            self.vars.append(bias_ii)

    def forward(self, x, vars=None):
        "Performs a forward pass on x, a numpy array of size (-1,shape[0])"
        if vars is None:
            vars = self.vars

        idx = 0
        for ii in range(len(self.shape)-1):
            w_ii, b_ii = vars[idx], vars[idx+1]
            x = F.linear(x, w_ii, b_ii)

            if self.ff_activation:
                x = F.relu(x, inplace=True)
            idx += 2

        w_ii, b_ii = vars[idx], vars[idx+1]
        x = F.linear(x, w_ii, b_ii)
        return x

class CNNet(torch.nn.Module):
    """PyTorch Module which implements a combined CNN-feedforward network for node classification.
    
    Attributes:
    conv_layers: ModuleList of Conv2d layers for CNN forward pass.
    ff_layers: ModuleList of Linear layers for feedforward pass.
    pool_layers: ModuleList of MaxPool2d layers for CNN forward pass. Contains Nones if no pooling.
    kernel: list of kernel sizes for CNN layers.
    stride: list of strides for CNN layers.
    padding: list of paddings for CNN layers.
    conv_activation: activation function to be applied between CNN layers
    ff_activation: activation function to be applied between feedforward layers.
    
    """
    def __init__(self,num_features,channels,ff_shape, input_size, kernel=2,stride=2, padding=0,
        conv_activation=None,ff_activation=None,pool=None):
        """Constructor for CNNet.

        Arguments:
            num_features: length of node feature vector.
            channels: vector of length N+1 specifying # of channels for each convolutional layer,
                where N is number of conv layers. channels[0] should be the size of the input image.
            ff_shape: vector specifying shape of feedforward network. ff_shape[0] should be 
                the size of the first hidden layer; constructor does the math to determine ff input size.
            input_size: tuple of input image size, (W1, H1)
            kernel: vector (or scalar) of kernel sizes for each conv layer. if scalar, each layer
                uses the same kernel.
            stride: vector (or scalar) of strides for each conv layer. uniform stride if scalar.
            padding: vector (or scalar) of paddings for each conv layer. uniform if scalar.
            conv_activation: nonlinear activation to be used after each conv layer
            ff_activation: nonlinear activation to be used after each ff layer
            pool: pooling to be added after each layer. if None, no pooling. if scalar, same pooling for each layer.
        """
        super().__init__()
        N = len(channels)-1 #number of conv layers
        if type(kernel) is int:
            self.kernel = [kernel]*N
        if type(stride) is int:
            self.stride = [stride]*N
        if type(padding) is int:
            self.padding = [padding]*N
        if not pool or len(pool)==1:
            self.pool = [pool]*N
        self.conv_activation = conv_activation
        self.ff_activation = ff_activation

        self.ff_shape = ff_shape
        self.channels = channels
        self.vars = nn.ParameterList()

        W, H = input_size
        for ii in range(0,len(channels)-1):
            W = int(1+(W-self.kernel[ii]+2*self.padding[ii])/self.stride[ii])
            H = int(1+(H-self.kernel[ii]+2*self.padding[ii])/self.stride[ii])

            weight_ii = nn.Parameter(torch.ones(channels[ii+1], channels[ii], self.kernel[ii], self.kernel[ii]))
            torch.nn.init.kaiming_normal_(weight_ii)
            self.vars.append(weight_ii)

            bias_ii = nn.Parameter(torch.ones(channels[ii+1]))
            self.vars.append(bias_ii)

            if self.pool[ii]:
                if W % self.pool[ii] != 0 or H % self.pool[ii] != 0:
                    raise ValueError('trying to pool by non-factor')
                W, H = W/self.pool[ii], H/self.pool[ii]

        cnn_output_size = W*H*channels[-1]+num_features
        shape = np.concatenate(([cnn_output_size], ff_shape))
        for ii in range(0,len(shape)-1):
            weight_ii = nn.Parameter(torch.ones(shape[ii+1], shape[ii]))
            torch.nn.init.kaiming_normal_(weight_ii)
            self.vars.append(weight_ii)

            bias_ii = nn.Parameter(torch.zeros(shape[ii+1]))
            self.vars.append(bias_ii)

    def forward(self, image_batch, feature_batch, vars=None):
        """Performs a network forward pass on images/features. Images go through CNN, these features are
            concatenated with the real-valued features, and passed through feed-forward network.
        
        Arguments:
        image_batch: batch of images (as a torch.Tensor of floats) to be passed through, of size [B,W1,H1,C1],
            where B is the batch_size, (W1,H1) and C1 are the input_size and channels[0] passed during initialization.
        feature_batch: batch of real-valued features (torch.Tensor of floats), of size [B,N], where N is the num_features
            passed during initialization.
            
        Usage: cnn = CNNet(...); outs = cnn(images_in,features_in)
        """
        if vars is None:
            vars = self.vars

        N = len(self.channels)-1 #number of conv layers

        x = image_batch
        idx = 0
        for ii in range(N):
            w_ii, b_ii = vars[idx], vars[idx+1]
            x = F.conv2d(x, w_ii, b_ii, stride=self.stride[ii], padding=self.padding[ii])
            if self.conv_activation:
                x = F.relu(x, inplace=True)
            if self.pool[ii]:
                x = F.max_pool2d(x, self.pool[ii], self.pool[ii], 0)
            idx += 2

        # flatten
        # x = x.view(x.size(0), -1)
        x = torch.flatten(x,start_dim=1)
        x = torch.cat((x,feature_batch),dim=1)

        for ii in range(len(self.ff_shape)-1):
            w_ii, b_ii = vars[idx], vars[idx+1]
            x = F.linear(x, w_ii, b_ii)

            if self.ff_activation:
                x = F.relu(x, inplace=True)
            idx += 2
        
        w_ii, b_ii = vars[idx], vars[idx+1]
        x = F.linear(x, w_ii, b_ii)
        return x
