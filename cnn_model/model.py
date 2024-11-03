import torch.nn as nn

class ChessModel(nn.Module):
    def __init__(self, num_classes):
        super(ChessModel, self).__init__()
        self.conv1 = nn.Conv2d(14, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)  # Batch Norm after conv1
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)  # Batch Norm after conv2
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(8 * 8 * 128, 256)
        self.bn3 = nn.BatchNorm1d(256)  # Batch Norm after fc1
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        # Initialize weights
        nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity='relu')
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)          # Apply Batch Norm
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)          # Apply Batch Norm
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.bn3(x)          # Apply Batch Norm
        x = self.relu(x)
        x = self.fc2(x)          # Output raw logits
        return x