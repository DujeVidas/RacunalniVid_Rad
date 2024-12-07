from PyTorchDataLoader_train import get_dataloader
import matplotlib.pyplot as plt

# Define the path to the HDF5 file
hdf5_file = 'train.h5'

# Create DataLoader
dataloader = get_dataloader(hdf5_file, batch_size=32, shuffle=True)

# Display 5 images (Both Bands)
def display_images(dataloader, num_images=5):
    # Get one batch from the dataloader
    images, labels = next(iter(dataloader))
    
    # Plot the images
    fig, axes = plt.subplots(2, num_images, figsize=(15, 6))  # Two rows: one for each band
    for i in range(num_images):
        img = images[i].numpy()  # Convert to numpy array
        label = labels[i].item()  # Get label as integer
        
        # Extract individual bands
        band_1 = img[0]  # First channel
        band_2 = img[1]  # Second channel

        # Determine the label description
        label_desc = "Iceberg" if label == 1 else "Ship"

        # Show band_1
        axes[0, i].imshow(band_1, cmap='gray')
        axes[0, i].set_title(f"Band 1 - {label_desc}")
        axes[0, i].axis('off')

        # Show band_2
        axes[1, i].imshow(band_2, cmap='gray')
        axes[1, i].set_title(f"Band 2 - {label_desc}")
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()

# Call the function
display_images(dataloader, num_images=5)
