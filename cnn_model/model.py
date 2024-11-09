import torch
import torch.nn as nn

class ChessModel(nn.Module):
    def __init__(self, num_classes):
        super(ChessModel, self).__init__()
        
        self.conv1 = nn.Conv2d(16, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)  # Batch Norm after conv1
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)  # Batch Norm after conv2
        
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(8 * 8 * 128, 256)
        self.bn3 = nn.BatchNorm1d(256)  # Batch Norm after fc1
        
        self.fc2 = nn.Linear(256, num_classes)
        
        self.relu = nn.ReLU()
        
        # Initialize weights
        self._initialize_weights()

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
        x = self.fc2(x)          
        x = torch.softmax(x, dim=1)  # Apply softmax activation
        return x
    
    def _initialize_weights(self):
        nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity='relu')
        nn.init.constant_(self.conv1.bias, 0)

        nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity='relu')
        nn.init.constant_(self.conv2.bias, 0)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0)

        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0)