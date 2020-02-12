# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 22:22:07 2020

@author: sabih
"""

from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(3,48,3,padding=1)
        self.conv2=nn.Conv2d(48,96,3,padding=1)
        self.conv3=nn.Conv2d(96,192,3,padding=1)
        self.pool=nn.MaxPool2d(2,2)
        self.dropout=nn.Dropout(0.25)
        self.fc1=nn.Linear(192*4*4,512)
        self.fc2=nn.Linear(512,256)
        self.fc3=nn.Linear(256,10)        
        
    def forward(self,x):
        x=self.pool(F.relu(self.conv1(x)))
        x=self.pool(F.relu(self.conv2(x)))
        x=self.pool(F.relu(self.conv3(x)))
        x=x.view(-1,192*4*4)
        x=self.dropout(x)
        x=F.relu(self.fc1(x))
        x=self.dropout(x)
        x=F.relu(self.fc2(x))
        x=self.dropout(x)
        x=F.relu(self.fc3(x))
        return x

def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    plt.imshow(np.transpose(img, (1, 2, 0))) 

num_workers = 0
batch_size = 20
valid_size = 0.2

transform=transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                              ])
train_data=datasets.CIFAR10('data',train=True,download=True,transform=transform)
test_data=datasets.CIFAR10('data',train=False,download=True,transform=transform)

n=len(train_data)
indices=list(range(n))
np.random.shuffle(indices)
split=int(np.floor(n*valid_size))
train_ind,valid_ind=indices[split:],indices[:split]

train_samp=SubsetRandomSampler(train_ind)
valid_samp=SubsetRandomSampler(valid_ind)

train_loader=torch.utils.data.DataLoader(train_data,sampler=train_samp,batch_size=batch_size,num_workers=num_workers)
valid_loader=torch.utils.data.DataLoader(train_data,sampler=valid_samp,batch_size=batch_size,num_workers=num_workers)
test_loader=torch.utils.data.DataLoader(test_data,batch_size=batch_size,num_workers=num_workers)

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck']

dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy()
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    imshow(images[idx])
    ax.set_title(classes[labels[idx]])

model=Network()
print(model)

criterian=nn.CrossEntropyLoss()
optimizer=optim.SGD(model.parameters(),lr=0.03)

epochs=5
valid_loss_min = np.Inf 
for i in range(epochs):
    train_loss = 0.0
    valid_loss = 0.0
    for image,label in train_loader:
        optimizer.zero_grad()
        output=model(image)
        loss=criterian(output,label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*image.size(0)
    model.eval()
    for image,label in valid_loader:
        output = model(image)
        loss = criterian(output, label)
        valid_loss += loss.item()*image.size(0)
        
    train_loss = train_loss/len(train_loader.sampler)
    valid_loss = valid_loss/len(valid_loader.sampler)
    
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epochs, train_loss, valid_loss))
    
    # save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        valid_loss_min,
        valid_loss))
        torch.save(model.state_dict(), 'model_augmented.pt')
        valid_loss_min = valid_loss