import torch
import abc
import numpy as np
import torch.nn.functional as F
from .base import Layer

class FuncConv(Layer):
    """
    Functional implementation of torch.nn.Conv2D to allow for
    network weights as task params.
    """
    def __init__(self, n_in, n_out, kernel, stride, padding, activation='relu'):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.kernel = kernel
        self.padding = padding
        self.stride = stride
        self.activation = self.get_activation(activation)

        self.weight = torch.nn.Parameter(torch.Tensor(kernel,kernel,n_in,n_out))
        self.bias = torch.nn.Parameter(torch.zeros(n_out))
        torch.nn.init.kaiming_uniform_(self.weight)
        stdv = 1 / np.sqrt(n_out)
        torch.nn.init.uniform_(self.bias, -stdv, stdv)

    def forward(self, x, z):
        weight, bias = z[0].type(x.dtype), z[1].type(x.dtype)

        # shape checks
        assert weight.shape[-4] == self.kernel
        assert weight.shape[-3] == self.kernel
        assert weight.shape[-2] == self.n_in
        assert weight.shape[-1] == self.n_out

        extra_indices = [ii for ii in range(len(weight.shape)-2)]

        batch_dim, c_, h_in, w_in = x.shape
        w_out = int(1+(w_in-self.kernel+2*self.padding)/self.stride)
        h_out = int(1+(h_in-self.kernel+2*self.padding)/self.stride)

        h_f, w_f, _, n_f = weight.shape
        output = torch.zeros(batch_dim, n_f, h_out, w_out).type(x.dtype)

        pd = (self.padding, self.padding)
        x_pad = F.pad(x, pd, "constant", 0.)
        _, _, h_in_pd, w_in_pd = x_pad.shape

        for idx_ii, ii in enumerate(range(0,h_in_pd-h_f,self.stride)):
            for idx_jj, jj in enumerate(range(0,w_in_pd-w_f,self.stride)):
                h_start, w_start = ii, jj
                h_end, w_end = h_start + h_f, w_start + w_f
                slc = x_pad[:, :, h_start:h_end, w_start:w_end]
                output[:,:,idx_ii,idx_jj] = torch.tensordot(slc, weight, dims=([1,2,3], [2,0,1])) + bias

        # TODO(acauligi): determine when activation should be applied at layer level
        # return self.activation(output)
        return output

    def task_params(self):
        return [self.weight, self.bias]

    def num_params(self):
        return 2

    def output_size(self):
        return self.n_out
