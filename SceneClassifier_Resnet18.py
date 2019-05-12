#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
import time
import os
import copy
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models

device = torch.device("cuda:0")


# In[2]:


class MyTrainDataset(Dataset):
    def __init__(self,csv_path):
        """
         Args:
            csv_path (string): csv 文件路径
            img_path (string): 图像文件所在路径
            transform: transform 操作
        """
        #Transformation
        self.to_tensor = transforms.ToTensor()
        self.Resize = transforms.RandomResizedCrop(224)
        self.Flip = transforms.RandomHorizontalFlip()
        #read the csv
        self.data_info = pd.read_csv(csv_path, header=None)
        #get the name
        self.image_arr = np.asarray(self.data_info.iloc[:,0])
        #get the label
        self.label_arr = np.asarray(self.data_info.iloc[:,1])
        #get the length
        self.data_length = len(self.data_info.index)
        
    def __getitem__(self, index):
        #get the file name from df
        single_image_name = self.image_arr[index]
        #get the data
        train_path = "cvdl2019/train_images/" + single_image_name
        img_data = Image.open(train_path)
        #Transformation
        imgtmp = self.Resize(img_data)
        imgflip = self.Flip(imgtmp)
        #turn the image into tensor
        img_as_tensor = self.to_tensor(imgflip).to(device)
        #GPU
        
        #get the label
        single_image_label = self.label_arr[index]
        
        return (img_as_tensor, single_image_label)
    def __len__(self):
        return self.data_length


# In[3]:


path = "cvdl2019/train_images/train_annotations.csv"
sceneset = MyTrainDataset(path)
labels_len = len(pd.read_csv('cvdl2019/scene_classes.csv'))+1
batch_size = 163
validation_split = .2
shuffle_dataset = True
random_seed = 42

#creating data indices for training and validation splits:
dataset_size = len(sceneset)

indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

#create PT data sampler and loaders
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(sceneset,batch_size=batch_size,sampler=train_sampler)
valid_loader = torch.utils.data.DataLoader(sceneset,batch_size=batch_size,sampler=valid_sampler)
dataloaders = {'train':train_loader,'val':valid_loader}
datasetSize = {'train':43104,'val':10775}


# In[4]:


loss = []
acc = []


# In[5]:


def train_model(model, criterion,optimizer,scheduler, num_epoch = 25):
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(num_epoch):
        print('Epoch {}/{}'.format(epoch, num_epoch - 1))
        print('*'*10)
        
        #each stage have training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()
                #print("debugging: train phase")
            else:
                model.eval()
                #print("debugging: eval phase")
            running_loss = 0.0
            running_corrects = 0
            
            #Iterate
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.cuda()
                labels = labels.cuda()
                #print("debugging: iter phase")
                #zero the parameter gradient
                #print("debugging: zero grad")
                optimizer.zero_grad()
                
                #forward
                with torch.set_grad_enabled(phase == 'train'):
                    #print("debugging: forward phase")
                    inputs = inputs.cuda()
                    outputs = model(inputs)
                    _, preds = torch.max(outputs,1)
                    loss = criterion(outputs,labels)
                    
                    #back
                    if phase == 'train':
                        #print("debugging: backward phase")
                        loss.backward()
                        optimizer.step()
                #stat
                running_loss += loss.item() *inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / datasetSize[phase]
            epoch_acc = running_corrects.double() / datasetSize[phase]
            loss.append(epoch_loss)
            acc.append(epoch_acc)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            
            #deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# In[6]:


model_ft = models.resnet18(pretrained=True)


# In[ ]:


#model_ft = models.resnet18(pretrained=True).to(device)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 80)

#model_ft = model_ft.cuda()

criterion = nn.CrossEntropyLoss()
model_ft = model_ft.to(device)
# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

optzer = optim.Adam(model_ft.parameters())
# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optzer, step_size=7, gamma=0.1)


# In[ ]:


model_ft.load_state_dict(torch.load('params.pkl'))


# In[ ]:


model_ft = train_model(model_ft, criterion, optzer,exp_lr_scheduler,
                       num_epoch = 20)


# In[ ]:


torch.save(model_ft.state_dict(),'model.pkl')


# In[ ]:


torch.save(model_ft.state_dict(),'params.pkl')


# In[ ]:


labels_len


# In[ ]:




