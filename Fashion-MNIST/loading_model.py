# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 18:30:24 2020

@author: sabih
"""

import torch
from torchvision import datasets, transforms
import matplotlib as plt
import torch.nn.functional as F
from torch import nn
from torch import optim

#class MyModel(nn.Module):
#    def __init__(self):
#        super().__init__()
#        self.hidden1=nn.Linear(784,512)
#        self.hidden2=nn.Linear(512,256)
#        self.hidden3=nn.Linear(256,128)
#        self.hidden4=nn.Linear(128,64)
#        self.output=nn.Linear(64,10)
#        
#    def forward(self,x):
#        x=x.view(x.shape[0],-1)
#        x=F.relu(self.hidden1(x))
#        x=F.relu(self.hidden2(x))
#        x=F.relu(self.hidden3(x))
#        x=F.relu(self.hidden4(x))
#        x=F.log_softmax(self.output(x),dim=1)
#        return x
#model=MyModel()

#checkpoint = {'input_size': 784,
#              'output_size': 10,
#              'hidden_layers': [each.out_features for each in model.hidden_layers],
#              'state_dict': model.state_dict()}
#
#torch.save(checkpoint, 'checkpoint.pth')

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
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
state_dict=torch.load('checkpoint.pth')
model.load_state_dict(state_dict)
#print(model.state_dict())

#def load_checkpoint(filepath):
#    checkpoint = torch.load(filepath)
#    model = MyModel(checkpoint['input_size'],
#                             checkpoint['output_size'],
#                             checkpoint['hidden_layers'])
#    model.load_state_dict(checkpoint['state_dict'])
#    
#    return model
#        
#
import helper
dataiter = iter(testloader)
images, labels = dataiter.next()
img = images[0]
img = img.resize_(1, 784)
ps = torch.exp(model(img))
helper.view_classify(img.resize_(1, 28, 28), ps, version='Fashion')    