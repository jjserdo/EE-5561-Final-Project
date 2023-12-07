"""
EE 5561 - Image Processing
Final Project
Deadline: December 17, 2023
coded by: Justine Serdoncillo start 12/6/23
"""

#### IMPORT LIBRARIES ####
import torch
from torchvision import datasets # maybe MS Coco detection
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
from net import *

#### LOAD DATASET ####

# visualize the data


#### SPLIT DATASET ####


#### SETUP MODEL ####
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = deepResUnet(3,1)
model.to(device)

# define the loss function
criterion = nn.CrossEntropyLoss()

# define the optimizer
learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# define epoch number
num_epochs = 10

# initialize the loss
loss_list = []
loss_list_mean = []

#### START TRAINING ####

#### VALIDATION #### ? how for image segmentation

#### VISUALIZE LOSS(?) ####

#### TEST MODEL ####

#### SAVE MODEL ####
fname = 
torch.save(model.state_dict(), fname + ".h5")

#### (OPTION) LOAD MODEL ####
fname = 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = deepResUnet()
model.to(device)
model.load_state_dict(torch.load(fname + ".h5"))
model.eval()

# test images