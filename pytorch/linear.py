import torch
import abc
import numpy as np
from .base import Layer

class FuncLinear(Layer):
    """
    Functional implementation of torch.nn.Linear to allow for
    network weights as task params.
    """
    def __init__(self, in_features, out_features, activation='none'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = self.get_activation(activation)
        self.weight = torch.nn.Parameter(torch.Tensor(out_features, 
                                                      in_features))
        self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        torch.nn.init.kaiming_uniform_(self.weight)
        stdv = 1 / np.sqrt(out_features)
        torch.nn.init.uniform_(self.bias, -stdv, stdv)

    def forward(self, x, z):
        weight, bias = z
        # shape checks
        assert weight.shape[-2] == self.out_features
        assert weight.shape[-1] == self.in_features
        assert (bias.shape[-1] == self.out_features) or (not bias)
        extra_indices = [ii for ii in range(len(weight.shape)-2)]
        output = x.matmul(weight.permute(*extra_indices, -1, -2))
        output += bias.unsqueeze(-2)

        # TODO(acauligi): determine when activation should be applied at layer level
        # return self.activation(output)
        return output

    def task_params(self):
        return [self.weight, self.bias]

    def num_params(self):
        return 2

    def output_size(self):
        return self.out_features
