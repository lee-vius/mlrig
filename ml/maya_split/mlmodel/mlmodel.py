import numpy as np
import torch
from torch import nn


class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_num, hidden_size, dropout=0.0, bn=False, pca_num=0):
        super().__init__()
        # The network can be used to train both differential and anchor points
        # Contain 2 Dense layers (input/output layer)
        # Contain #layers_num hidden dense layers
        # Contain a PCA layer at the end of network

        # Read in the parameters of Network initialization
        self.input_size = input_size # The size of input feature
        self.output_size = output_size # The size of output feature
        self.hidden_num = hidden_num # The number of hidden layers
        self.hidden_size = hidden_size # The out put size of hidden layers
        self.pca_num = pca_num # The number of principal components that will be kept

        # Construct the input layer
        self.input_fc = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU()
        )

        # Construct the hidden, batchnorm and dropout layers
        self.hidden_layers = []
        self.activation_layers = []
        self.batchnorm_layers = []
        self.dropout_layers = []
        for i in range(hidden_num):
            self.hidden_layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            if bn:
                self.batchnorm_layers.append(nn.BatchNorm1d(self.hidden_size))
            self.activation_layers.append(nn.ReLU())
            if dropout > 0.01:
                self.dropout_layers.append(nn.Dropout(p=dropout))
        
        # Construct the output layers
        self.output_fc = nn.Linear(self.hidden_size, self.output_size)


    def forward(self, x):
        x = self.input_fc(x)
        for i in range(self.hidden_num):
            x = self.hidden_layers[i](x)
            if self.batchnorm_layers:
                x = self.batchnorm_layers[i](x)
            x = self.activation_layers[i](x)
            if self.dropout_layers:
                x = self.dropout_layers[i](x)

        x = self.output_fc(x)
        # Apply PCA
        if self.pca_num != 0:
            components = torch.tensor(self.PCA_svd(x, self.pca_num), dtype=torch.float32)
            x = torch.mm(components.t(), x - components.mean())
            x = torch.mm(components, x) + components.mean()
        return x

    @staticmethod
    def PCA_svd(X, k, center=True):
        n = X.size()[0]
        ones = torch.ones(n).view([n,1])
        h = ((1/n) * torch.mm(ones, ones.t())) if center  else torch.zeros(n*n).view([n,n])
        H = torch.eye(n) - h
        X_center =  torch.mm(H.double(), X.double())
        u, s, v = torch.svd(X_center)
        components  = v[:k].t()
        #explained_variance = torch.mul(s[:k], s[:k])/(n-1)
        return components
