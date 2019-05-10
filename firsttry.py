#!/usr/bin/env python
# coding: utf-8

# In[144]:


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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[93]:


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
        img_as_tensor = self.to_tensor(imgflip)
        
        #get the label
        single_image_label = self.label_arr[index]
        
        return (img_as_tensor, single_image_label)
    def __len__(self):
        return self.data_length


# In[188]:



path = "cvdl2019/train_images/train_annotations.csv"
sceneset = MyTrainDataset(path)
labels_len = len(pd.read_csv('cvdl2019/scene_classes.csv'))
batch_size = 100
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


# In[ ]:


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
                print("debugging: train phase")
            else:
                model.eval()
                print("debugging: eval phase")
            running_loss = 0.0
            running_corrects = 0
            
            #Iterate
            for inputs, labels in dataloaders[phase]:
                #inputs = inputs.to(device)
                #labels = labels.to(device)
                print("debugging: iter phase")
                #zero the parameter gradient
                print("debugging: zero grad")
                optimizer.zero_grad()
                
                #forward
                with torch.set_grad_enabled(phase == 'train'):
                    print("debugging: forward phase")
                    outputs = model(inputs)
                    _, preds = torch.max(outputs,1)
                    loss = criterion(outputs,labels)
                    
                    #back
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                #stat
                running_loss += loss.item() *inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / datasetSize[phase]
            epoch_acc = running_corrects.double() / datasetSize[phase]
            
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


# In[159]:


def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


# In[206]:


model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
optzer = optim.Adam(model.parameters())
# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


# In[207]:


model_ft = train_model(model, criterion, optzer, exp_lr_scheduler,
                       num_epoch = 25)


# In[182]:


iter_1 = iter(dataloaders['train'])
features, labels = next(iter_1)
features.shape, labels.shape


# In[196]:


model = models.vgg16(pretrained=True)
for param in model.parameters():
    param.requires_grad = False


# In[197]:


n_classes = 80
n_inputs = model.classifier[6].in_features
model.classifier[6] = nn.Sequential(
                      nn.Linear(n_inputs,256),nn.ReLU(),nn.Dropout(0.4),nn.Linear(256,n_classes),nn.LogSoftmax(dim = 1))


# In[ ]:




