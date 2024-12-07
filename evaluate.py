import torch
import matplotlib.pyplot as plt
from PyTorchDataLoader_test import get_dataloader
from model import IcebergClassifierCNN

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Model
model = IcebergClassifierCNN().to(device)
model.load_state_dict(torch.load("iceberg_classifier.pth"))
model.eval()

# Load Data
hdf5_file = 'test.h5'
dataloader = get_dataloader(hdf5_file, batch_size=32, shuffle=False)

# Evaluation Loop for Unlabeled Data
predictions = []
images_to_plot = []

with torch.no_grad():
    for images in dataloader:  # Skip labels during loading
        images = images.to(device)
        outputs = model(images)
        predicted = (outputs > 0.5).float()
        predictions.extend(predicted.cpu().numpy())  # Collect predictions
        images_to_plot.extend(images.cpu().numpy())  # Collect images for plotting

# Display 10 images and their predictions
num_to_display = 10
fig, axes = plt.subplots(2, num_to_display, figsize=(15, 6))  # Two rows: one for each band

for i in range(min(num_to_display, len(images_to_plot))):
    image = images_to_plot[i]  # Shape: (2, 75, 75)
    prediction = predictions[i]
    category = "Iceberg" if prediction == 1 else "Ship"

    # Extract individual bands
    band_1 = image[0]  # First channel
    band_2 = image[1]  # Second channel

    # Show band_1
    axes[0, i].imshow(band_1, cmap='gray')
    axes[0, i].set_title(f"Band 1 - {category}")
    axes[0, i].axis('off')

    # Show band_2
    axes[1, i].imshow(band_2, cmap='gray')
    axes[1, i].set_title(f"Band 2 - {category}")
    axes[1, i].axis('off')

plt.tight_layout()
plt.show()
