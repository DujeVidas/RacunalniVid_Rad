import torch
from torch.utils.data import DataLoader
from load_data import load_data
from model import SimpleCNN

def validate_model(val_loader, model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, Y_batch in val_loader:
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            total += Y_batch.size(0)
            correct += (predicted == Y_batch).sum().item()
    return correct / total

if __name__ == "__main__":
    file_path = "input_data.npz"
    _, val_dataset = load_data(file_path)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = SimpleCNN()
    model.load_state_dict(torch.load("simple_cnn.pth"))

    accuracy = validate_model(val_loader, model)
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")
