import numpy as np
from torch import nn


class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_num, hidden_size, dropout=0.0, bn=False):
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
        return x
