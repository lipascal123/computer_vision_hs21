import torch
import torch.nn as nn
from torch.nn.modules.activation import ReLU
from torch.nn.modules.linear import Linear


class LinearClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.codename = 'linear'

        # Define the network layers in order.
        # Input is 2D.
        # Output is a single value.
        # Single linear layer.
        self.layers = nn.Sequential(
            nn.Linear(in_features=2, out_features=1, bias=True)
        )
    
    def forward(self, batch):
        # Process batch using the layers.
        x = self.layers(batch)
        # Final sigmoid activation to obtain a probability.
        return torch.sigmoid(x)


class MLPClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.codename = 'mlp'

        # Define the network layers in order.
        # Input is 2D.
        # Output is a single value.
        # Multiple linear layers each followed by a ReLU non-linearity (apart from the last).
        #raise NotImplementedError()
        self.layers = nn.Sequential(
            # 2 hidden, each 16 neurons, followed by ReLU activation     
            # 1 prediciton layer,
            nn.Linear(2,16),
            nn.ReLU(),
            nn.Linear(16,16),
            nn.ReLU(),
            nn.Linear(16,1)
        )
    
    def forward(self, batch):
        # Process batch using the layers.
        x = self.layers(batch)
        # Final sigmoid activation to obtain a probability.
        return torch.sigmoid(x)
