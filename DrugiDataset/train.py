import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from load_data import load_data
from model import SimpleCNN

def train_model(train_loader, model, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        for X_batch, Y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

if __name__ == "__main__":
    file_path = "input_data.npz"
    train_dataset, _ = load_data(file_path)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = SimpleCNN()
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    
    train_model(train_loader, model, criterion, optimizer)
    torch.save(model.state_dict(), "simple_cnn.pth")
