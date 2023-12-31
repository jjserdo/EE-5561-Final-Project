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

#### LOAD DATASET ####
orig_imag_path = r"C:\Users\rickl\Desktop\Camouflage\Images\Train"  #Using the animal images here
imag_segmented_path = r"C:\Users\rickl\Desktop\Camouflage\GT"       #Using the mask images here
test_images = r"C:\Users\rickl\Desktop\Camouflage\Images\Test"      #Using the test images here
orig_imag_path = "..\data\Camouflage\images" 
imag_segmented_path =  "..\data\Camouflage\mask"

# We need to make sure that the files will be organized in the same way in both folders for matching them:
image_list = sorted(os.listdir(orig_imag_path))
mask_list = sorted(os.listdir(imag_segmented_path))
#test_image_list = sorted(os.listdir(test_images))

# We'll transform it into a ToTensor and the images have different sizes, so we need to resize. I set it as 256.
transform = v2.Compose([
    v2.Resize((224, 224)),
    v2.Compose([v2.ToPILImage(), v2.ToTensor()]),
])

data_full = []
# Now let's transform all of our images in the ToTensor type as required for pytorch:
#for i in range(10):
for i in range(len(image_list)-1):
    img_name = os.path.join(orig_imag_path, image_list[i])
    mask_name = os.path.join(imag_segmented_path, mask_list[i])
    image = transform(Image.open(img_name).convert("RGB"))
    mask =  transform(Image.open(mask_name).convert("L"))

    data_full.append((image,mask))

#test_data = []
#for i in range(10):
#for i in range(len(test_image_list)-1):
#    img_name_test = os.path.join(test_images, test_image_list[i])
#    image_test = transform(Image.open(img_name_test).convert("RGB"))


#    test_data.append(image_test)

# visualize the data
#The data_full explained: data_full[0][0] --> first [0] is the image being observed (lines) ; second [0] is the column, indicates if we are looking the image (0) or the mask (1).
plt.imshow(data_full[0][0].permute(1, 2, 0))  # when we transform int ToTensor the order is different than the one used in Imshow, that's why we use  permute here.
plt.axis('off')
plt.show()

### SPLIT DATASET ####
# train, test, validation separation

data_full = data_full[0:50]
train_size = int(0.8 * len(data_full))  #Let's use 80% of the data as training and 20% as validation --> we can adjust as we want...
valid_size = len(data_full) - train_size

train_data = data_full[:train_size]
valid_data = data_full[train_size:]

batch_size = 50
#batch_size = 40

train_loader = torch.utils.data.DataLoader(train_data,
                                                batch_size = batch_size,
                                                shuffle=True),
#          'test': torch.utils.data.DataLoader(test_data,
#                                              batch_size = batch_size,
#                                               shuffle=True),
           'valid': torch.utils.data.DataLoader(valid_data,
                                                batch_size = batch_size,
                                                shuffle=True)

# plt.imshow(data2[0][0].permute(1, 2, 0))
# plt.axis('off')
# plt.show()


### SETUP MODEL ####
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = deepResUnet(3,1).to(device)

# define the loss function
criterion = nn.MSELoss()


# define the optimizer
learning_rate = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# define epoch number
num_epochs = 30

# initialize the loss
loss_list = []
loss_list_mean = []

#### START TRAINING ####
# Stochastic gradient descent used in training.
# The Deep residual U-Net paper uses the mean squared error the loss function.
#  L(W) = 1/N * Sum||Net(Ii,W) - s_i||^2
# Where N is the number of training samples.
iter = 0
for epoch in range(num_epochs):
    print('Epoch: {}'.format(epoch))
    loss_buff = []
    for images,masks in train_loader:
        # getting the images and labels from the training dataset
        images = images.to(device)
        masks = masks.to(device)

        # clear the gradients
        optimizer.zero_grad()

        # call the NN
        outputs = model(images)
        #print(f"shape of outputs: {outputs.shape}")
        #print(f"shape of masks: {masks.shape}")
        # loss calculation
        loss = criterion(outputs, masks)
        loss_buff = np.append(loss_buff, loss.item())

        # back propagation
        loss.backward()

        loss_list = np.append(loss_list, (loss_buff))

        # update parameters
        optimizer.step()

        iter += 1

        if iter % 10 == 0:
            print('Iterations: {}'.format(iter))

#### VALIDATION #### ? how for image segmentation

        if iter % 100 == 0:

            # accuracy
            correct = 0
            total = 0

            for (images, labels) in valid_loader:
                # getting the images and labels from the training dataset
                images = images.to(device)
                labels = labels.to(device)

                # clear the gradients
                optimizer.zero_grad()

                # call the NN
                outputs = model(images)

                # get the predictions
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)

                correct += (predicted == labels).sum()

            accuracy = 100 * correct / total

            print('Iterations: {} Loss: {}. Validation Accuracy: {}'.
                  format(iter, loss.item(), accuracy))

        loss_list_mean = np.append(loss_list_mean, (loss.item()))
        ################################

# visualize the loss
plt.plot(loss_list)
plt.plot(loss_list_mean)


#### VISUALIZE LOSS(?) ####

#### TEST MODEL ####
model.eval()  # Set the model to evaluation mode

correct = 0
total = 0

# Iterate through the test DataLoader
for (images, masks) in test_loader:
    # Getting the images and masks from the test dataset
    images = images.to(device)
    masks = masks.to(device)

    # No need to clear gradients or call optimizer.zero_grad() during testing

    # Call the NN
    outputs = model(images)

    # Assuming your model outputs probabilities, you can convert them to class predictions
    predicted = (outputs > 0.5).float()  # Binary threshold for binary segmentation

    # Update total count
    total += masks.numel()  # Total number of pixels

    # Update correct count by comparing predicted and ground truth masks
    correct += (predicted == masks).sum().item()

#### SAVE MODEL ####
# fname = "weights"
# torch.save(model.state_dict(), fname + ".h5")
