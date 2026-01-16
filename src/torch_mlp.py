import torch
import torch.nn as nn


class TorchMLP(nn.Module):
    """
    MLP with a single hidden layer, implemented in PyTorch.

    Architecture: Re-create your MLP in PyTorch
        Input are 2 dimensions
          - Linear layer from input to hidden
          - Sigmoid activation
          - Linear layer towards output
          - Sigmoid activation (or might use softmax later-on)
          - Output (3)

    Notes:
    - The forward pass already returns sigmoid outputs (probabilities).
    - Loss function and optimizer are defined outside this class.
    - Backpropagation is handled automatically by PyTorch's autograd.
    """
    def __init__(self, hidden_dim=5):
        """
        Parameters:
            hidden_dim : int
                Number of neurons in the hidden layer.
        """
        super().__init__()

        self.fc1 = nn.Linear(2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass of the network.

        x: Tensor of shape (N, 2)

        returns:
            y_hat: Tensor of shape (N, 3)
                   Sigmoid outputs (class-wise probabilities)
        """
        z1 = self.sigmoid(self.fc1(x))
        y_hat = self.sigmoid(self.fc2(z1))
        return y_hat