# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 17:58:22 2020

@author: sabih
"""

# import libraries
import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch import optim,nn
import torch.nn.functional as F

num_workers = 0
batch_size = 20
transform = transforms.ToTensor()
train_data = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
test_data = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
    num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 
    num_workers=num_workers)

#    
## obtain one batch of training images
#dataiter = iter(train_loader)
#images, labels = dataiter.next()
#images = images.numpy()
#
## plot the images in the batch, along with the corresponding labels
#fig = plt.figure(figsize=(25, 4))
#for idx in np.arange(20):
#    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
#    ax.imshow(np.squeeze(images[idx]), cmap='gray')
#    # print out the correct label for each image
#    # .item() gets the value contained in a Tensor
#    ax.set_title(str(labels[idx].item()))
#    
#    
#img = np.squeeze(images[1])
#
#fig = plt.figure(figsize = (12,12)) 
#ax = fig.add_subplot(111)
#ax.imshow(img, cmap='gray')
#width, height = img.shape
#thresh = img.max()/2.5
#for x in range(width):
#    for y in range(height):
#        val = round(img[x][y],2) if img[x][y] !=0 else 0
#        ax.annotate(str(val), xy=(y,x),
#                    horizontalalignment='center',
#                    verticalalignment='center',
#                    color='white' if img[x][y]<thresh else 'black')



## TODO: Define the NN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # linear layer (784 -> 1 hidden node)
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2=nn.Linear(512,512)
        self.fc3=nn.Linear(512,10)

    def forward(self, x):
        # flatten image input
        x = x.view(-1, 28 * 28)
        # add hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        x = F.dropout(x,p=0.2)
        x = F.relu(self.fc2(x))
        x = F.dropout(x,p=0.2)
        x = self.fc3(x)
        
        return x

model = Net()
print(model)

n_epochs = 30
model.train()
criterion = nn.CrossEntropyLoss()
optimizer=optim.SGD(model.parameters(),lr=0.03)
for epoch in range(n_epochs):
    train_loss = 0.0
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*data.size(0)
        
    train_loss = train_loss/len(train_loader.sampler)

    print('Epoch: {} \tTraining Loss: {:.6f}'.format(
        epoch+1, 
        train_loss
        ))
    
    