import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
# Replace 'file.npz' with your .npz file's path
file_path = 'input_data.npz'

# Load the .npz file
data = np.load(file_path)

"""
# List all the keys (names of arrays stored in the file)
print("Keys in the .npz file:", data.files)

# Access and print each array
for key in data.files:
    print(f"Array name: {key}")
    print(data[key])
    print()  # For readability
"""
# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(data['X_train'], dtype=torch.float32).permute(0, 3, 1, 2)  # Channels-first format
Y_train_tensor = torch.tensor(data['Y_train'], dtype=torch.long)
X_val_tensor = torch.tensor(data['X_validation'], dtype=torch.float32).permute(0, 3, 1, 2)
Y_val_tensor = torch.tensor(data['Y_validation'], dtype=torch.long)

# Create PyTorch datasets and loaders
train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, Y_val_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Define a simple CNN model
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

# Initialize model, loss, and optimizer
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
    model.train()
    for X_batch, Y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, Y_batch)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Validation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for X_batch, Y_batch in val_loader:
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        total += Y_batch.size(0)
        correct += (predicted == Y_batch).sum().item()

print("Validation Accuracy:", correct / total)

# Close the file after use
data.close()