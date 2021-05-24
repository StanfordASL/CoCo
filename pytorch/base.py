import torch
import abc
class Layer(torch.nn.Module, abc.ABC):
    """
    Abstract class implementing the Layer interface. Used in Flex Architecture
    """
    def __init__(self):
        super().__init__()
    @abc.abstractmethod
    def forward(self, x, z):
        """
        Abstract forward pass for inputs x, task variables z.
        Note: should be able to handle the case when x has shape [B, *, n_x],
        where B is the batch dim, * are arbitrary dimensions, and n_x is
        the input dimension.
        """
        pass
    @property
    @abc.abstractmethod
    def task_params(self):
        pass
    @property
    @abc.abstractmethod
    def num_params(self):
        """
        Helper function to get number of task params required by
        forward pass.
        """
        pass
    @property
    @abc.abstractmethod
    def output_size(self):
        """
        Helper function to get the number of channels in the output
        of this layer. Used to determine the next layers input size
        """
        pass

    @staticmethod
    def get_activation(activation):
        """
        Function that returns a callable implementing an activation
        based on the string name of the activation
        """
        if activation == 'tanh':
            return torch.nn.Tanh()
        elif activation == 'relu':
            return torch.nn.ReLU()
        elif activation == 'sin':
            return torch.sin
        elif activation == 'none':
            return torch.nn.Identity()
        else:
            raise ValueError("unknown activation" + activation)
