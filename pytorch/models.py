import torch
import numpy as np

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
        super(FFNet, self).__init__()
        self.shape = shape
        self.layers = []
        self.activation = activation ##TODO(pculbertson): make it possible use >1 activation... maybe? who cares
        for ii in range(0,len(shape)-1):
            self.layers.append(torch.nn.Linear(shape[ii],shape[ii+1]))

        self.layers = torch.nn.ModuleList(self.layers)

    def forward(self, x):
        "Performs a forward pass on x, a numpy array of size (-1,shape[0])"
        for ii in range(0,len(self.layers)-1):
            x = self.layers[ii](x)
            if self.activation:
              x = self.activation(x)

        return self.layers[-1](x)

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
        super(CNNet, self).__init__()
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

        self.conv_layers = []
        self.pool_layers = []
        self.ff_layers = []
        W, H = input_size
        for ii in range(0,len(channels)-1):
            self.conv_layers.append(torch.nn.Conv2d(channels[ii],channels[ii+1],self.kernel[ii],
                stride=self.stride[ii],padding=self.padding[ii]))
            W = int(1+(W-self.kernel[ii]+2*self.padding[ii])/self.stride[ii])
            H = int(1+(H-self.kernel[ii]+2*self.padding[ii])/self.stride[ii])
            if self.pool[ii]:
                if W % self.pool[ii] != 0 or H % self.pool[ii] != 0:
                    raise ValueError('trying to pool by non-factor')
                W, H = W/self.pool[ii], H/self.pool[ii]
                self.pool_layers.append(torch.nn.MaxPool2d(self.pool[ii]))
            else:
                self.pool_layers.append(None)

        cnn_output_size = W*H*channels[-1]+num_features
        shape = np.concatenate(([cnn_output_size], ff_shape))
        for ii in range(0,len(shape)-1):
            self.ff_layers.append(torch.nn.Linear(shape[ii],shape[ii+1]))

        self.conv_layers = torch.nn.ModuleList(self.conv_layers)
        self.ff_layers = torch.nn.ModuleList(self.ff_layers)
        if pool:
            self.pool_layers = torch.nn.ModuleList(self.pool_layers)
    
    def forward(self, image_batch, feature_batch):
        """Performs a network forward pass on images/features. Images go through CNN, these features are
            concatenated with the real-valued features, and passed through feed-forward network.
        
        Arguments:
        image_batch: batch of images (as a torch.Tensor of floats) to be passed through, of size [B,W1,H1,C1],
            where B is the batch_size, (W1,H1) and C1 are the input_size and channels[0] passed during initialization.
        feature_batch: batch of real-valued features (torch.Tensor of floats), of size [B,N], where N is the num_features
            passed during initialization.
            
        Usage: cnn = CNNet(...); outs = cnn(images_in,features_in)
        """
        x = image_batch
        for ii in range(0,len(self.conv_layers)):
            x = self.conv_layers[ii](x)
            if self.conv_activation:
                x = self.conv_activation(x)
            if self.pool_layers[ii]:
                x = self.pool_layers[ii](x)

        x = torch.flatten(x,start_dim=1)
        x = torch.cat((x,feature_batch),dim=1)
        for ii in range(0,len(self.ff_layers)-1):
            x = self.ff_layers[ii](x)
            if self.ff_activation:
                x = self.ff_activation(x)
        
        return self.ff_layers[-1](x)
