'''
Justine Serdoncillo
EE 5561 - Image Processing
Problem Set 4
December 6, 2023
'''

# %% Problem Statement
"""
In this exercise, you will implement a ResNet structure and will
use it for MNIST database classification. You can use the given PyTorch tutorial for training
and replace the model with ResNet by using the following ResNet9.
"""


# %% import libraries
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np

# %% LOAD THE MNIST DATASET
data_full = datasets.MNIST(root = 'data',
                           train = True,
                           transform = ToTensor(),
                           download = True)

# visualize the data
plt.imshow(data_full.data[0])
plt.imshow(data_full.data[0], cmap='gray')
plt.imshow(data_full.data[61], cmap='gray')

# access the data
data1 = data_full.data[0]
plt.imshow(data1, cmap='gray')

# visualize multiple images
figure = plt.figure(figsize=(10,10))
cols, rows = 5,5

for i in range(1, cols*rows+1):
    
    idx = torch.randint(len(data_full), size=(1,)).item()
    
    img, label = data_full[idx]
    
    figure.add_subplot(rows,cols,i)
    plt.title('Number: ' + str(label))
    plt.axis('off')
    plt.imshow(img.squeeze(), cmap='gray')
    
plt.show()


# %% SPLIT THE DATASET
# train, test, validation seperation
train_data, test_data, valid_data = torch.utils.data.random_split(data_full, [30000, 10000, 20000])

batch_size = 100

loaders = {'train': torch.utils.data.DataLoader(train_data, 
                                                batch_size = batch_size, 
                                                shuffle=True),
           'test': torch.utils.data.DataLoader(test_data, 
                                                batch_size = batch_size, 
                                                shuffle=True),
           'valid': torch.utils.data.DataLoader(valid_data, 
                                                batch_size = batch_size, 
                                                shuffle=True)}
"""
#visualize the dictionary
train_part = loaders.get('train')
data2 = train_part.dataset
element1 = data2[0][0].squeeze()
plt.imshow(element1, cmap='gray')
"""


# %% 
##########################
## EE 5561 Assignment 4 ##
## JJ Personal Code     ##
##########################
def conv_block(in_channels, out_channels, pool=False, res=False, pool_size=2):
    layers = []
    if res:
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
    else:    
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3))
    layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.ReLU())
    if pool:
        layers.append(nn.MaxPool2d(pool_size))
    return nn.Sequential(*layers)
        

class ResNet9(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        # Preparation Layer
        self.prep = conv_block(in_channels, 32)
        # Layer 1 w/ Res Blocks
        self.layer1_conv = conv_block(32, 64, pool=True)
        self.layer1_res  = nn.Sequential(conv_block(64, 64, res=True),
                                         conv_block(64, 64, res=True))
        # Layer 2
        self.layer2 = conv_block(64, 128)
        # Layer 1 w/ Res Blocks
        self.layer3_conv = conv_block(128, 256, pool=True)
        self.layer3_res  = nn.Sequential(conv_block(256, 256, res=True),
                                         conv_block(256, 256, res=True))
        # Classifier
        self.classifier = nn.Sequential(nn.MaxPool2d(2),
                                        nn.Flatten(),
                                        nn.Linear(1024,10))
                                        #nn.Linear(1024,256),
                                        #nn.ReLU(),
                                        #nn.Linear(256,64),
                                        #nn.ReLU(),
                                        #nn.Linear(64,10),
                                        #nn.ReLU())
        self.shape_tester = nn.Sequential(nn.MaxPool2d(2),
                                        nn.Flatten())
        self.fc = nn.Linear(1024,10)
        
        # Params
        self.num_classes = num_classes
        
    def class3(self, size1d, num_classes):
        return nn.Linear(size1d, num_classes)
        
    def forward(self, x):
        
        # Preparation Layer
        y = self.prep(x)
        
        # Layer 1 w/ Res Blocks
        y1 = self.layer1_conv(y)
        #print(f"Residual in layer 1 has size: {y1.shape}") # has 12,12
        #y = self.layer1_res(y1) + y1
        y2 = self.layer1_res(y1)
        #print(f"After Residual in layer 1 has size: {y2.shape}") # has 8,8
        y = y1 + y2
        
        # Layer 2
        y = self.layer2(y)
        
        # Layer 3 w/ Res Blocks
        y1 = self.layer3_conv(y)
        y = self.layer3_res(y1) + y1
        
        # Classifier 
        y = self.shape_tester(y)
        #print(f"After Flatten in classifier layer has size: {y.shape}")
        y = self.fc(y)
        
        return y

# %% SET UP THE MODEL 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNet9(1, 10)
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


# %% TRANING STARTS HERE

iter = 0
for epoch in range(num_epochs):
    
    print('Epoch: {}'.format(epoch))
    
    loss_buff = []
    
    for i, (images,labels) in enumerate(loaders['train']):
        
        # getting the images and labels from the training dataset
        images = images.requires_grad_().to(device)
        labels = labels.to(device)
        
        # clear the gradients
        optimizer.zero_grad()
    
        # call the NN
        outputs = model(images) # torch.Size([100, 1, 28, 28])
        
        # loss calculation
        loss = criterion(outputs, labels)
        loss_buff = np.append(loss_buff, loss.item())
        
        # back propagation
        loss.backward()
        
        loss_list = np.append(loss_list, (loss_buff))
        
        #update parameters
        optimizer.step()
        
        iter += 1
        
        if iter % 10 == 0:
            print('Iterations: {}'.format(iter))

####### VALIDATION PART#############

        if iter % 100 == 0:
            
            # accuracy
            correct = 0
            total = 0
            
            for i, (images,labels) in enumerate(loaders['valid']):
                
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
        
#visualize the loss
plt.plot(loss_list)
plt.plot(loss_list_mean)

fig, ax = plt.subplots()
ax.plot(loss_list)
ax.set_title("Loss List using ResNet9")
ax.set_xlabel("Images")
ax.set_ylabel("Loss")

fig, ax = plt.subplots()
ax.plot(loss_list_mean)
ax.set_title("Loss List Mean using ResNet9")
ax.set_xlabel("Iterations")
ax.set_ylabel("Loss")
ax.plot(loss_list_mean)

# %% TEST PART#############
correct = 0
total = 0
            
for i, (images,labels) in enumerate(loaders['test']):
                
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
            
print('Iterations: {} Loss: {}. Test Accuracy: {}'.format(iter, loss.item(), accuracy))

# %%

# Save:
torch.save(model.state_dict(), "weights_1Linear.h5")


# Load:
device = torch.device('cpu')
model = ResNet9(1, 10)
model.to(device)
model.load_state_dict(torch.load("weights_1Linear.h5"))
model.eval()
# visualize multiple images
fig, ax = plt.subplots(5,5,figsize=(10,10))
fig.suptitle("Some Predictions of Trained Model on test ")
cols, rows = 5,5

for i in range(5):
    for j in range(5):
        for test_images, test_labels in loaders['test']:  
            img = test_images[0]  
            img.unsqueeze_(1)
        img = img.to(device)
        optimizer.zero_grad()
        #print(img.shape)
        outputs = model(img)
        _, predicted = torch.max(outputs.data, 1)
        
        ax[i,j].set_title('Number: ' + str(predicted.numpy()[0]))
        ax[i,j].axis('off')
        ax[i,j].imshow(img.squeeze().numpy(), cmap='gray')
