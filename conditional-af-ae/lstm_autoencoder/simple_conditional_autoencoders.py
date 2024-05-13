from dataclasses import dataclass
import torch
import torch.nn as nn
from collections import OrderedDict

@dataclass
class ParamsIO:
    """
    This class allows to define the input and output of each neural network
    """
    input_dim_beta: int
    output_dim_beta: int
    input_dim_factor: int
    output_dim_factor: int

class extract_tensor(nn.Module):
    def forward(self,x):
        # Output shape (batch, features, hidden)
        tensor, _ = x
        # Reshape shape (batch, hidden)
        return tensor[:, :, :]

class CA:
    """
    This class allows to define methods which are commune to conditional autoencoders define in the paper
    """

    @staticmethod
    def _factor_network(input_dim_factor, output_dim_factor):
        return nn.Linear(input_dim_factor, output_dim_factor)

    @staticmethod
    def _forward(beta_network, x_beta: torch.Tensor, factor_network, x_factor: torch.Tensor):
        return beta_network(x_beta) @ factor_network(x_factor)


class CA0(nn.Module):
    """
    First autoencoder of the paper
    Uses a single linear layer in both the beta and factor networks
    Can be compared to a PCA
    No activation needed
    """

    def __init__(
            self,
            params: ParamsIO = None
    ):
        super().__init__()
        self.params = params
        self.beta_network = nn.Linear(self.params.input_dim_beta, self.params.output_dim_beta)
        self.factor_network = CA._factor_network(self.params.input_dim_factor, self.params.output_dim_factor)

    def forward(self, x_beta, x_factor):
        return CA._forward(self.beta_network, x_beta, self.factor_network, x_factor)


class CA1(nn.Module):
    """
    Second autoencoder of the paper
    Difference between above: one hidden layer in the beta network, therefore using relu activation
    """

    def __init__(
            self,
            params: ParamsIO = None,
            config: dict = None
    ):
        super().__init__()
        self.params = params
        self.rate_dropout = config['rate_dropout']  # Dropout do not figure in the paper, but can be useful
        self.activation = nn.ReLU()
        self.beta_network_base = [('hidden1', nn.Linear(self.params.input_dim_beta, 32)),
                                  ('dropout1', nn.Dropout(self.rate_dropout)),
                                  ('batchnorm1', nn.BatchNorm1d(2)),
                                  ('relu1', self.activation)]
        self.beta_network = nn.Sequential(
            OrderedDict(self.beta_network_base + [('linear1', nn.Linear(32, self.params.output_dim_beta))]))
        self.factor_network = CA._factor_network(self.params.input_dim_factor, self.params.output_dim_factor)

    def forward(self, x_beta, x_factor):
        return CA._forward(self.beta_network, x_beta, self.factor_network, x_factor)


class CA2(nn.Module):
    """
    Third autoencoder of the paper
    Difference between above: one more hidden layer in the beta network
    """

    def __init__(
            self,
            params: ParamsIO = None,
            config: dict = None
    ):
        super().__init__()
        self.params = params
        self.rate_dropout = config['rate_dropout']
        self.activation = nn.ReLU()
        self.beta_network_base = CA1(self.params, config).beta_network_base + \
                                 [('hidden2', nn.Linear(32, 16)),
                                  ('dropout2', nn.Dropout(self.rate_dropout)),
                                  ('batchnorm2', nn.BatchNorm1d(2)),
                                  ('relu2', self.activation)]
        self.beta_network = nn.Sequential(
            OrderedDict(self.beta_network_base + [('linear1', nn.Linear(16, self.params.output_dim_beta))]))
        self.factor_network = CA._factor_network(self.params.input_dim_factor, self.params.output_dim_factor)

    def forward(self, x_beta, x_factor):
        return CA._forward(self.beta_network, x_beta, self.factor_network, x_factor)


class CA3(nn.Module):
    """
    Third autoencoder of the paper
    Difference between above: one more hidden layer in the beta network
    """

    def __init__(
            self,
            params: ParamsIO = None,
            config: dict = None
    ):
        super().__init__()
        self.params = params
        self.rate_dropout = config['rate_dropout']
        self.activation = nn.ReLU()
        self.beta_network_base = CA2(self.params, config).beta_network_base + \
                                 [('hidden3', nn.Linear(16, 8)),
                                  ('dropout3', nn.Dropout(self.rate_dropout)),
                                  ('batchnorm3', nn.BatchNorm1d(2)),
                                  ('relu3', self.activation)]
        self.beta_network = nn.Sequential(
            OrderedDict(self.beta_network_base + [('linear1', nn.Linear(8, self.params.output_dim_beta))]))
        self.factor_network = CA._factor_network(self.params.input_dim_factor, self.params.output_dim_factor)

    def forward(self, x_beta, x_factor):
        return CA._forward(self.beta_network, x_beta, self.factor_network, x_factor)