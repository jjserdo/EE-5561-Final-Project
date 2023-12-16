"""
EE 5561 - Image Processing
Final Project
Deadline: December 17, 2023
coded by: Justine Serdoncillo start 12/6/23
adapated by: Ricardo Linhares
"""

#### IMPORT LIBRARIES ####
import torch
from torchvision import datasets # maybe MS Coco detection
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
from PIL import Image
from net import *
import os
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

#### UPLOAD TEST IMAGES ####
images_path = "..\data\Camouflage\images"
labels_path =  "..\data\Camouflage\mask"
image_list = sorted(os.listdir(images_path))
mask_list = sorted(os.listdir(labels_path))

transform = v2.Compose([
    v2.Resize((224, 224)),
    v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
])

data_full = []
# Now let's transform all of our images in the ToTensor type as required for pytorch:
for i in range(10):
    img_name = os.path.join(images_path, image_list[i])
    mask_name = os.path.join(labels_path, mask_list[i])
    image = transform(Image.open(img_name).convert("RGB"))
    mask =  transform(Image.open(mask_name).convert("L"))
    data_full.append((image,mask))
    
test_data = data_full[:]
batch_size = 2
loaders = {'test': torch.utils.data.DataLoader(test_data,
                                               batch_size = batch_size,
                                               shuffle=True)}

#### LOADING SAVED WEIGHTS #### 
fname = "weights"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = deepResUnet(3, 1)
model.to(device)
model.load_state_dict(torch.load(fname + ".h5"))
model.eval()
learning_rate = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# visualize multiple images
rows, cols = 3,3
fig, ax = plt.subplots(rows,cols,figsize=(10,10))
fig.suptitle("Some Predictions of Trained Model on test ")

for i in range(rows):
    for test_images, test_labels in loaders['test']:  
        img = test_images[0]
        label = test_labels[0]
        
    ax[i, 0].set_title("Original Photo")
    ax[i, 0].imshow(img.permute(1, 2, 0))
    ax[i, 0].axis('off')
    
    img.unsqueeze_(0)     
    #print(img.shape)
    img = img.to(device)
    optimizer.zero_grad()
    output = model(img)
    output = torch.squeeze(output, 0)

    ax[i, 1].set_title("Actual Mask Photo")
    ax[i, 1].imshow(label.permute(1, 2, 0), cmap="gray")
    ax[i, 1].axis('off')
    
    ax[i, 2].set_title("Predicted Mask Photo")
    ax[i, 2].imshow(output.detach().permute(1, 2, 0), cmap="gray")
    ax[i, 2].axis('off')
    