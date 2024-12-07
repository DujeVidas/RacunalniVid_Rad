import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np

class IcebergDataset(Dataset):
    def __init__(self, hdf5_file):
        self.hdf5_file = hdf5_file
        with h5py.File(self.hdf5_file, 'r') as f:
            self.keys = list(f.keys())  # Get all entry keys

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        with h5py.File(self.hdf5_file, 'r') as f:
            entry = f[self.keys[idx]]
            band_1 = np.array(entry['band_1'])  # (75, 75)
            band_2 = np.array(entry['band_2'])  # (75, 75)
        
        # Stack bands to create a 2-channel image
        image = np.stack([band_1, band_2], axis=0)  # Shape: (2, 75, 75)
        return torch.tensor(image, dtype=torch.float32)

def get_dataloader(hdf5_file, batch_size=32, shuffle=True):
    dataset = IcebergDataset(hdf5_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)