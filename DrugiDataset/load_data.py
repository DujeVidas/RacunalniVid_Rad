import numpy as np
import torch
from torch.utils.data import TensorDataset

def load_data(file_path):
    # Load the .npz file
    data = np.load(file_path)
    
    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(data['X_train'], dtype=torch.float32).permute(0, 3, 1, 2)  # Channels-first
    Y_train_tensor = torch.tensor(data['Y_train'], dtype=torch.long)
    X_val_tensor = torch.tensor(data['X_validation'], dtype=torch.float32).permute(0, 3, 1, 2)
    Y_val_tensor = torch.tensor(data['Y_validation'], dtype=torch.long)
    
    # Create PyTorch datasets
    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, Y_val_tensor)
    
    return train_dataset, val_dataset

if __name__ == "__main__":
    file_path = "input_data.npz"
    train_dataset, val_dataset = load_data(file_path)
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
