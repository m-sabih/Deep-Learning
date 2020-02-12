# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 16:04:13 2020

@author: sabih
"""

import torch
from torchvision import datasets, transforms
import matplotlib as plt
from torch import nn
from torch import optim
import numpy as np

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
# Download and load the training data
trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load the test data
testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

model= nn.Sequential(nn.Linear(784,512),
                     nn.ReLU(),
                     nn.Dropout(p=0.2),                     
                     nn.Linear(512,256),
                     nn.ReLU(),
                     nn.Dropout(p=0.2),
                     nn.Linear(256,128),
                     nn.ReLU(),
                     nn.Dropout(p=0.2),
                     nn.Linear(128,64),
                     nn.ReLU(),
                     nn.Dropout(p=0.2),
                     nn.Linear(64,10),
                     nn.LogSoftmax(dim=1))
criterian=nn.NLLLoss()
optimizer=optim.SGD(model.parameters(),lr=0.03)
epochs=5
train_losses=[]
test_losses=[]
for i in range(epochs):
    running_loss=0
    for images,label in trainloader:
        optimizer.zero_grad()
        images=images.view(images.shape[0],-1)
        output=model(images)
        loss=criterian(output,label)
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
    else:
        test_loss = 0
        accuracy = 0
        with torch.no_grad():
            model.eval()
            for images,labels in testloader:
                images=images.view(images.shape[0],-1)
                log_ps=model(images)
                ps=torch.exp(log_ps)
                test_loss += criterian(log_ps, labels)
                prob,class_ind=ps.topk(1,dim=1)
                equals=class_ind==labels.view(class_ind.shape)
                accuracy+=torch.mean(equals.type(torch.FloatTensor))
        #print(f'Accuracy: {accuracy.item()*100}%')    
        model.train()
        train_losses.append(running_loss/len(trainloader))
        test_losses.append(test_loss/len(testloader))
        print("Epoch: {}/{}.. ".format(i+1, epochs),
          "Training Loss: {:.3f}.. ".format(running_loss/len(trainloader)),
          "Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
          "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))
        
#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'
#
#import matplotlib.pyplot as plt 
#import numpy as np 
#x = np.linspace(-10 , 10, 100)
#y = np.sin(x) 
#plt.plot(train_losses, label='Training loss')
#plt.plot(test_losses, label='Validation loss')
#plt.legend(frameon=False)

#import helper
#model.eval()
#dataiter = iter(testloader)
#images, labels = dataiter.next()
#img = images[0]
#img = img.resize_(1, 784)
#with torch.no_grad():
    #ps = torch.exp(model(img))
#helper.view_classify(img.resize_(1, 28, 28), ps, version='Fashion')
    
    
#saving model
#torch.save(model.state_dict(),'checkpoint.pth')
#checkpoint = {'input_size': 784,
#              'output_size': 10,
#              'hidden_layers': [512,256,128,64],
#              'state_dict': model.state_dict()}
#
#torch.save(checkpoint, 'checkpoint.pth')