import torch.nn as nn
import torch.nn.functional as F
import torch

class IcebergClassifierCNN(nn.Module):
    def __init__(self):
        super(IcebergClassifierCNN, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1)  # 2 channels for bands
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)

        # Dynamically calculate the input size for fc1
        dummy_input = torch.zeros(1, 2, 75, 75)
        dummy_output = self._forward_conv(dummy_input)
        flattened_size = dummy_output.view(1, -1).size(1)

        self.fc1 = nn.Linear(flattened_size, 256)
        self.fc2 = nn.Linear(256, 1)

    def _forward_conv(self, x):
        """Helper method to pass data through convolutional layers."""
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        print(f"Shape before fc1: {x.shape}")  # Debugging
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))  # Sigmoid for binary classification
        return x
