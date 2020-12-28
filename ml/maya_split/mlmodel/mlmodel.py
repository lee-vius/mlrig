import numpy as np
import torch
from torch import nn


class Network_diff(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, dropout=0.0):
        super().__init__()
        # The network can be used to train both differential and anchor points
        # Contain 2 Dense layers (input/output layer)
        # Contain 5 hidden dense layers
        # Contain a PCA layer at the end of network

        # Read in the parameters of Network initialization
        self.input_size = input_size # The size of input feature
        self.output_size = output_size # The size of output feature
        self.hidden_size = hidden_size # The out put size of hidden layers

        # Construct the input layer
        self.input_fc = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU()
        )

        self.hidden1 = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

        self.hidden2 = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

        self.hidden3 = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

        self.hidden4 = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

        self.hidden5 = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )
        # Construct the output layers
        self.output_fc = nn.Linear(self.hidden_size, self.output_size)


    def forward(self, x):
        x = self.input_fc(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.hidden4(x)
        x = self.output_fc(x)
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


class Network_anchor(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, dropout=0.0):
        super().__init__()
        # The network can be used to train both differential and anchor points
        # Contain 2 Dense layers (input/output layer)
        # Contain 3 hidden dense layers

        # Read in the parameters of Network initialization
        self.input_size = input_size # The size of input feature
        self.output_size = output_size # The size of output feature
        self.hidden_size = hidden_size # The out put size of hidden layers

        # Construct the input layer
        self.input_fc = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU()
        )

        self.hidden1 = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

        self.hidden2 = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

        self.hidden3 = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )
        # Construct the output layers
        self.output_fc = nn.Linear(self.hidden_size, self.output_size)


    def forward(self, x):
        x = self.input_fc(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.output_fc(x)
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