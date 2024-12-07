import torch.nn as nn
import torch

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(64 * 17 * 17, 128)  # Adjust based on input size
        self.fc2 = nn.Linear(128, 2)  # Assuming binary classification

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # Apply ReLU after conv1
        x = self.pool(torch.relu(self.conv2(x)))  # Apply ReLU after conv2
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))  # Apply ReLU after fc1
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    model = SimpleCNN()
    print(model)
