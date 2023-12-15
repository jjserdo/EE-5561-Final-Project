# Testing for relative path for file uploading 

#### IMPORT LIBRARIES ####
import torch
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

TF_ENABLE_ONEDNN_OPTS=0

transform = v2.Compose([
    v2.Resize((256, 256)),
    v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
])

print(os.getcwd())

orig_imag_path = "..\data\Camouflage\Images\Train" 
image_list = sorted(os.listdir(orig_imag_path))
i = 0
img_name = os.path.join(orig_imag_path, image_list[i])
image = transform(Image.open(img_name).convert("RGB"))

plt.imshow(image.permute(1, 2, 0))
