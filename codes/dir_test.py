# Testing for relative path for file uploading 

#### IMPORT LIBRARIES ####
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

transform = v2.Compose([
    v2.Resize((256, 256)),
    v2.ToTensor(),
])

print(os.getcwd())

orig_imag_path = "..\data\Camouflage\Images\Train" 
image_list = sorted(os.listdir(orig_imag_path))
i = 0
img_name = os.path.join(orig_imag_path, image_list[i])
image = transform(Image.open(img_name).convert("RGB"))

plt.imshow(image)
