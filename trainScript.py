import torch
import torch.optim as optim
import torch.nn as nn
from PyTorchDataLoader_train import get_dataloader
from model import IcebergClassifierCNN

# Hyperparameters
batch_size = 32
learning_rate = 0.001
num_epochs = 10
hdf5_file = 'train.h5'

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Data
dataloader = get_dataloader(hdf5_file, batch_size=batch_size, shuffle=True)

# Initialize Model
model = IcebergClassifierCNN().to(device)
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device).float()
        labels = labels.unsqueeze(1)  # Add a dimension for BCE Loss

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}")

# Save the Model
torch.save(model.state_dict(), "iceberg_classifier.pth")
