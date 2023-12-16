import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion, binary_dilation

# Import your deep residual U-Net implementation
from net_1 import *

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

        return image, mask
    
# Check if GPU is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate the U-Net model and move it to the GPU
in_channels = 3  # Adjust based on your dataset
out_channels = 1  # Adjust based on your task (e.g., binary segmentation)
model = DeepResUNet(in_channels, out_channels).to(device)

# Define loss function and optimizer
optimizer = optim.Adam(model.parameters(),lr=1e-4)
criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy for binary segmentation

# Create datasets and data loaders
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Adjust size as needed
    transforms.ToTensor(),
])

# Instantiate the dataset and DataLoader
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

dataset = CustomDataset(root=r"C:\Users\sheno\OneDrive\Documents\PIV\CAMO-COCO-V.1.0\CAMO-COCO-V.1.0-CVIU2019\Camouflage", transform=transform)

# Split the dataset into train, validation, and test
train_size = int(0.85 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        # print(f"Output Size: {outputs.size()}, Target Size: {targets.size()}")
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs, targets).item()

    val_loss /= len(val_loader)
    print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss}')

# Test the model on the test set
model.eval()
test_loss = 0.0
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)  # Move tensors to the same device as the model
        outputs = model(inputs)
        test_loss += criterion(outputs, targets).item()

test_loss /= len(test_loader)
print(f'Test Loss: {test_loss}')

# Post-processing
def post_process(mask):
    # Apply binary erosion and dilation for refinement
    eroded_mask = binary_erosion(mask)
    post_processed_mask = binary_dilation(eroded_mask)
    return post_processed_mask

# Plot 5 examples from the test set
test_examples = []

for inputs, targets in test_loader:
    inputs, targets = inputs.to(device), targets.to(device)  # Move tensors to the same device as the model
    with torch.no_grad():
        outputs = model(inputs)

    test_examples.extend(list(zip(inputs.cpu(), targets.cpu(), outputs.cpu())))  # Move tensors back to CPU for plotting

plt.figure(figsize=(15, 10))

for i, (input_image, target_mask, output_mask) in enumerate(test_examples[:5], 1):
    input_image = transforms.ToPILImage()(input_image.squeeze(0))
    target_mask = target_mask.cpu().numpy().squeeze()
    output_mask = torch.sigmoid(output_mask).cpu().numpy().squeeze()

    # Apply post-processing to the predicted mask
    post_processed_output = post_process(output_mask)

    plt.subplot(5, 3, 3 * i - 2)
    plt.imshow(input_image)
    plt.title(f"Example {i}: Input Image")
    plt.axis("off")

    plt.subplot(5, 3, 3 * i - 1)
    plt.imshow(target_mask, cmap="gray")
    plt.title(f"Example {i}: Target Mask")
    plt.axis("off")

    plt.subplot(5, 3, 3 * i)
    plt.imshow(output_mask, cmap="gray")
    plt.title(f"Example {i}: Predicted Mask")
    plt.axis("off")

plt.tight_layout()
plt.show()
