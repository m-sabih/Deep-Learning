# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 00:01:08 2020

@author: sabih
"""

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
test_data=datasets.CIFAR10('data',train=False,download=True,transform=transform)
test_loader=torch.utils.data.DataLoader(test_data,batch_size=batch_size,num_workers=num_workers,shuffle=True)

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck']

model=Network()
state_dict=torch.load('model_augmented.pt')
model.load_state_dict(state_dict)
#print(model)
# track test loss
test_loss = 0.0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
criterion=nn.CrossEntropyLoss()
optimizer=optim.SGD(model.parameters(),lr=0.03)
model.eval()
#for data, target in test_loader:
#    output = model(data)
#    loss = criterion(output, target)
#    test_loss += loss.item()*data.size(0)
#    _, pred = torch.max(output, 1)    
#    correct_tensor = pred.eq(target.data.view_as(pred))
#    correct = np.squeeze(correct_tensor.cpu().numpy())
#    for i in range(batch_size):
#        label = target.data[i]
#        class_correct[label] += correct[i].item()
#        class_total[label] += 1
#
## average test loss
#test_loss = test_loss/len(test_loader.dataset)
#print('Test Loss: {:.6f}\n'.format(test_loss))
#for i in range(10):
#    if class_total[i] > 0:
#        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
#            classes[i], 100 * class_correct[i] / class_total[i],
#            np.sum(class_correct[i]), np.sum(class_total[i])))
#    else:
#        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))
#print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
#    100. * np.sum(class_correct) / np.sum(class_total),
#    np.sum(class_correct), np.sum(class_total)))
#testing
dataiter = iter(test_loader)
images, labels = dataiter.next()
#images, labels = dataiter.next()
#images, labels = dataiter.next()
images.numpy()
# get sample outputs
output = model(images)
# convert output probabilities to predicted class
_, preds_tensor = torch.max(output, 1)
preds = np.squeeze(preds_tensor.cpu().numpy())
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    imshow(images.cpu()[idx])
    ax.set_title("{} ({})".format(classes[preds[idx]], classes[labels[idx]]),
                 color=("green" if preds[idx]==labels[idx].item() else "red"))