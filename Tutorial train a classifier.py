#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import torchvision
import torchvision.transforms as tfs


# In[3]:


transform = tfs.Compose([
    tfs.ToTensor(),
    tfs.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

trainset = torchvision.datasets.CIFAR10(root = './data',train=True,download=True,transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size = 4,
                                          shuffle = True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root = './data',train = False,
                                       download = True, transform = transform)
testloader = torch.utils.data.DataLoader(testset,batch_size = 4,
                                         shuffle = False, num_workers = 2)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog','frog', 'horse'
           'ship', 'truck')


# In[4]:


import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img = img/2 +0.5 #unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()
    
#get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

#show images
imshow(torchvision.utils.make_grid(images))
#print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


# In[5]:


import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    


# In[6]:


net = Net()


# In[7]:


#define a Loss Function

import torch.optim as opt

criterion = nn.CrossEntropyLoss()
optimizer = opt.SGD(net.parameters(), lr=0.001,momentum = 0.9)


# In[8]:


#train the network

for epoch in range(2):
    
    running_loss = 0
    for i, data in enumerate(trainloader, 0):
        #get the input
        inputs, labels = data
        
        #zero the parameter gradients
        optimizer.zero_grad()
        
        #forward + backward + optimize
        output = net(inputs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        #print stat
        running_loss += loss.item()
        if i % 2000 == 1999:#every 2000 mini-batches
            print('[%d,%5d] loss: %.3f' %
                 (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
print("Finish Training")


# In[13]:


#Test on the test data

dataiter = iter(testloader)
images, labels = dataiter.next()

#print
imshow(torchvision.utils.make_grid(images))
print('Ground Truth: ',' '.join('%5s' % classes[labels[j]] for j in range(4)))


# In[19]:


outputs = net(images)


# In[20]:


_, predicted = torch.max(outputs,1)
print('Predicted: ', " ".join('%5s' % classes[predicted[j]] for j in range(4)) )


# In[21]:


correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the network on the 10000 test images: %d %%' %(
100 * correct / total))


# In[23]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print( device)


# In[ ]:




