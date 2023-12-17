import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion, binary_dilation
import numpy as np
# Import your deep residual U-Net implementation
# from net_1 import *

class CustomDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images = sorted(os.listdir(os.path.join(root, 'Images')))
        self.masks = sorted(os.listdir(os.path.join(root, 'GT')))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, 'Images', self.images[idx])
        mask_path = os.path.join(self.root, 'GT', self.masks[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        # Convert the mask to binary
        mask = (mask > 0).float()

        return image, mask

# Check if GPU is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate the U-Net model and move it to the GPU
in_channels = 3  # Adjust based on your dataset
out_channels = 1  # Adjust based on your task (e.g., binary segmentation)
model = DeepResUNet(in_channels, out_channels).to(device)

# Define loss function and optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy for binary segmentation

# Create datasets and data loaders
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Adjust size as needed
    transforms.ToTensor(),
])

dataset = CustomDataset(root=r"/content/drive/MyDrive/Camouflage", transform=transform)

# Split the dataset into train, validation, and test
train_size = int(0.85 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

num_epochs = 100

# Early stopping parameters
patience = 10
early_stopping_counter = 0
best_val_loss = float('inf')

# Training loop
for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)['output']
        targets = (targets > 0).float()

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)['output']
            targets = (targets > 0).float()
            val_loss += criterion(outputs, targets).item()

    val_loss /= len(val_loader)
    print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss}')

    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= patience:
            print(f'Early stopping at epoch {epoch+1} as validation loss did not improve.')
            break

# Test the model on the test set and print accuracy
model.eval()
correct_predictions = 0
total_predictions = 0

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)['output']
        targets = (targets > 0).float()

        # Apply threshold to predicted masks
        predicted_masks = torch.sigmoid(outputs)
        predicted_masks_binary = (predicted_masks > 0.5).float()

        # Compute accuracy
        correct_predictions += (predicted_masks_binary == targets).sum().item()
        total_predictions += targets.numel()

test_accuracy = correct_predictions / total_predictions
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

# Plot 5 examples from the test set
test_examples = []

for inputs, targets in test_loader:
    inputs, targets = inputs.to(device), targets.to(device)  # Move tensors to the same device as the model
    with torch.no_grad():
        outputs = model(inputs)['output']

    # Move tensors back to CPU for plotting
    input_images_cpu = inputs.cpu()
    target_masks_cpu = (targets > 0).float().cpu()  # Ensure that targets are binary masks
    output_masks_cpu = torch.sigmoid(outputs).cpu()

    test_examples.extend(list(zip(input_images_cpu, target_masks_cpu, output_masks_cpu)))

plt.figure(figsize=(15, 10))

for i, (input_image, target_mask, output_mask) in enumerate(test_examples[:5], 1):
    input_image = transforms.ToPILImage()(input_image.squeeze(0))
    target_mask = target_mask.numpy().squeeze()
    output_mask = output_mask.numpy().squeeze()

    plt.subplot(5, 4, 4 * i - 3)
    plt.imshow(input_image)
    plt.title(f"Example {i}: Input Image")
    plt.axis("off")

    plt.subplot(5, 4, 4 * i - 2)
    plt.imshow(target_mask, cmap="gray")
    plt.title(f"Example {i}: Target Mask")
    plt.axis("off")

    plt.subplot(5, 4, 4 * i-1)
    plt.imshow(output_mask, cmap="gray")  # Display the post-processed output
    plt.title(f"Example {i}: Predicted Mask ")
    plt.axis("off")

    # Overlay Detected Camouflage on Input Image
    overlay = np.copy(input_image)
    overlay[output_mask > 0.5] = [255, 0, 0]  # Highlight detected camouflage regions in red

    plt.subplot(5, 4, 4 * i)
    plt.imshow(overlay)
    plt.title(f"Example {i}: Overlay")
    plt.axis("off")

plt.tight_layout()
plt.show()
