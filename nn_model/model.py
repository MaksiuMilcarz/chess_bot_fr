import torch.nn as nn
import torch.nn.functional as F

class ChessModel(nn.Module):
    def __init__(self, num_classes):
        """
        Initialize the ChessModel with convolutional and fully connected layers,
        including batch normalization for improved training.

        Args:
        - num_classes (int): Number of unique move classes.
        """
        super(ChessModel, self).__init__()
        # Convolutional layers with batch normalization
        self.conv1 = nn.Conv2d(in_channels=13, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        # Fully connected layers with batch normalization
        self.fc1 = nn.Linear(in_features=8 * 8 * 128, out_features=256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(in_features=256, out_features=num_classes)

        # Activation function
        self.relu = nn.ReLU()

        # Weight initialization
        nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity='relu')
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        """
        Define the forward pass of the model.

        Args:
        - x (torch.Tensor): Input tensor of shape (batch_size, 13, 8, 8).

        Returns:
        - x (torch.Tensor): Output logits of shape (batch_size, num_classes).
        """
        # Convolutional Layer 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # Convolutional Layer 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        # Flatten
        x = x.view(-1, 8 * 8 * 128)

        # Fully Connected Layer 1
        x = self.fc1(x)
        x = self.bn3(x)
        x = self.relu(x)

        # Output Layer
        x = self.fc2(x)

        return x