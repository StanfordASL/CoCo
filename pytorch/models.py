import pdb
import torch
import numpy as np
from    torch import nn
from    torch.nn import functional as F

from .base import Layer
from .linear import FuncLinear
from .conv import FuncConv

class FFNet(torch.nn.Module):
    """
    Implements a generic feed-forward neural network, with
    conitioning options.
    """
    def __init__(self, net_shape, activation='relu', cond_type='all_weights'):
        """Constructor for FFNet.
        
        Arguments:
            net_shape: list of ints describing network shape, including input & output size.
            activation: a torch.nn function specifying the network activation.
        """
        super().__init__()
        self.net_shape = net_shape

        if activation == 'tanh':
            self.activation = torch.nn.Tanh()
        elif activation == 'relu':
            self.activation = torch.nn.ReLU()
        else:
            raise ValueError("unknown activation" + activation)

        if cond_type in ['all_weights', 'none']:
            self.cond_type = cond_type
        else:
            raise ValueError("unknown cond_type" + cond_type)

        self.layers = []
        self.z0 = []
        for ii in range(0,len(net_shape)-1):
            if self.cond_type == 'all_weights':
                self.layers.append(FuncLinear(
                    self.net_shape[ii],self.net_shape[ii+1]))
                self.z0 += self.layers[-1].task_params()
            elif self.cond_type == 'none':
                # add a linear layer
                self.layers.append(torch.nn.Linear(self.net_shape[ii],
                                                   self.net_shape[ii+1]))

        self.layers = torch.nn.ModuleList(self.layers)
        self.z0 = torch.nn.ParameterList(self.z0)

    def forward(self, x, task_params):
        """
        Performs a 'conditioned' forward pass through the Architecture.
        Maps inputs x, task_params to outputs, according to specific
        conditioning layers, etc.
        """
        if self.cond_type == 'none':
            # We ignore task parameters if there is no conditioning set. This is
            # usually because Learner was set to Overfit.
            assert task_params is None

        tp_ind = 0
        for ii in range(len(self.layers) - 1):
            if self.cond_type == 'none':
                x = self.layers[ii](x)
            elif self.cond_type == 'all_weights':
                x = self.layers[ii](x, task_params[tp_ind:tp_ind+2])
                tp_ind += self.layers[ii].num_params()
            x = self.activation(x)

        # output layer
        if self.cond_type == 'all_weights':
            x = self.layers[-1](x, task_params[tp_ind:])
        else:
            x = self.layers[-1](x)
        return x

    def prior(self, batch_size):
        """
        Returns a list of prior task_parameters, expanded to batch_size.
        """
        return [zz.clone().reshape(1, *zz.shape).expand(
            batch_size, *zz.shape) for zz in self.z0]

class CNNet(torch.nn.Module):
    def __init__(self,num_features,channels,ff_shape, input_size, kernel=2,stride=2, padding=0,
        conv_activation='relu',ff_activation='relu',pool=None, cond_type='mixed_weights'):
        super().__init__()

        self.channels = channels
        N = len(channels)-1

        if type(kernel) is int:
            self.kernel = [kernel]*N
        if type(stride) is int:
            self.stride = [stride]*N
        if type(padding) is int:
            self.padding = [padding]*N
        if not pool or len(pool)==1:
            self.pool = [pool]*N

        if ff_activation == 'tanh':
            self.ff_activation = torch.nn.Tanh()
        elif ff_activation == 'relu':
            self.ff_activation = torch.nn.ReLU()
        else:
            raise ValueError("unknown ff_activation" + ff_activation)

        if conv_activation == 'tanh':
            self.conv_activation = torch.nn.Tanh()
        elif conv_activation == 'relu':
            self.conv_activation = torch.nn.ReLU()
        else:
            raise ValueError("unknown conv_activation" + conv_activation)

        if cond_type in ['all_weights', 'mixed_weights', 'none']:
            self.cond_type = cond_type
        else:
            raise ValueError("unknown cond_type" + cond_type)

        ws, bs = [], []
        W, H = input_size

        self.z0 = []
        self.conv_layers = []
        self.pool_layers = []
        self.ff_layers = []
        for ii in range(N):
            if self.cond_type == 'all_weights':
                self.conv_layers.append(FuncConv(
                    channels[ii], channels[ii+1], self.kernel[ii], self.stride[ii], self.padding[ii]))

                self.z0 += self.conv_layers[-1].task_params()
            elif self.cond_type == 'mixed_weights':
                self.conv_layers.append(torch.nn.Conv2d(channels[ii],channels[ii+1],self.kernel[ii],
                    stride=self.stride[ii],padding=self.padding[ii]))

                w_ii, b_ii = list(self.conv_layers[-1].parameters())
                self.z0 += [w_ii, b_ii]
            elif self.cond_type == 'none':
                self.conv_layers.append(torch.nn.Conv2d(channels[ii],channels[ii+1],self.kernel[ii],
                    stride=self.stride[ii],padding=self.padding[ii]))

            if self.pool[ii]:
                if W % self.pool[ii] != 0 or H % self.pool[ii] != 0:
                    raise ValueError('trying to pool by non-factor')
                W, H = W/self.pool[ii], H/self.pool[ii]
                self.pool_layers.append(torch.nn.MaxPool2d(self.pool[ii]))
            else:
                self.pool_layers.append(None)

            W = int(1+(W-self.kernel[ii]+2*self.padding[ii])/self.stride[ii])
            H = int(1+(H-self.kernel[ii]+2*self.padding[ii])/self.stride[ii])

        cnn_output_size = W*H*channels[-1]+num_features
        self.ff_shape = np.concatenate(([cnn_output_size], ff_shape))

        for ii in range(0,len(self.ff_shape)-1):
            if self.cond_type in ['all_weights', 'mixed_weights']:
                self.ff_layers.append(FuncLinear(
                    self.ff_shape[ii],self.ff_shape[ii+1]))
                self.z0 += self.ff_layers[-1].task_params()
            elif self.cond_type == 'none':
                # add a linear layer
                self.ff_layers.append(torch.nn.Linear(self.ff_shape[ii],
                                                   self.ff_shape[ii+1]))

        self.conv_layers = torch.nn.ModuleList(self.conv_layers)
        self.ff_layers = torch.nn.ModuleList(self.ff_layers)
        self.z0 = torch.nn.ParameterList(self.z0)
        if pool:
            self.pool_layers = torch.nn.ModuleList(self.pool_layers)

    def forward(self, image_batch, feature_batch, task_params):
        if self.cond_type == 'none':
            # We ignore task parameters if there is no conditioning set. This is
            # usually because Learner was set to Overfit.
            assert task_params is None

        x = image_batch
        tp_ind = 0
        for ii in range(len(self.conv_layers)):
            if self.cond_type == 'none':
                x = self.conv_layers[ii](x)
            elif self.cond_type == 'all_weights':
                x = self.conv_layers[ii](x, task_params[tp_ind:tp_ind+2])
                tp_ind += self.conv_layers[ii].num_params()
            elif self.cond_type == 'mixed_weights':
                w_ii, b_ii = task_params[tp_ind:tp_ind+2]
                x = F.conv2d(x, w_ii, b_ii, stride=self.stride[ii], padding=self.padding[ii])
                tp_ind += len(list(self.conv_layers[ii].parameters()))
            x = self.conv_activation(x)
            if self.pool_layers[ii]:
                x = self.pool_layers[ii](x)

        x = torch.flatten(x,start_dim=1)
        x = torch.cat((x,feature_batch),dim=1)

        for ii in range(len(self.ff_layers) - 1):
            if self.cond_type == 'none':
                x = self.ff_layers[ii](x)
            elif self.cond_type in ['all_weights', 'mixed_weights']:
                x = self.ff_layers[ii](x, task_params[tp_ind:tp_ind+2])
                tp_ind += self.ff_layers[ii].num_params()
            x = self.ff_activation(x)

        # output layer
        if self.cond_type in ['all_weights', 'mixed_weights']:
            x = self.ff_layers[-1](x, task_params[tp_ind:])
        else:
            x = self.ff_layers[-1](x)
        return x 

    def prior(self, batch_size):
        """
        Returns a list of prior task_parameters, expanded to batch_size.
        """
        return [zz.clone().reshape(1, *zz.shape).expand(
            batch_size, *zz.shape) for zz in self.z0]
