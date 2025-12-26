"""Simple CNN model for MNIST classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTNet(nn.Module):
    """Simple CNN for MNIST.
    
    Architecture:
    - Conv2d(1, 32, 3) -> ReLU -> MaxPool2d(2)
    - Conv2d(32, 64, 3) -> ReLU -> MaxPool2d(2)
    - Linear(1600, 128) -> ReLU -> Dropout
    - Linear(128, 10)
    
    Total parameters: ~206,922
    """
    
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv block 1
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        
        # Conv block 2
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


def create_model() -> nn.Module:
    """Create and initialize the MNIST model."""
    model = MNISTNet()
    # Initialize weights
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    return model
