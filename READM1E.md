

```python
import sys
sys.path.append('..')
from PIL import Image
from torchvision import transforms as tfs
```


```python
im = Image.open("cat.jpg")
im
```




![png](output_1_0.png)




```python
#Resize
print('Before scale, shape: {}'.format(im.size))
new_im = tfs.Resize((100,200))(im)
print('After scale, shape: {}'.format(new_im.size))
new_im
```

    Before scale, shape: (250, 308)
    After scale, shape: (200, 100)





![png](output_2_1.png)




```python
#Random Crop
random_im = tfs.RandomCrop(100)(im)#100*100
random_im
```




![png](output_3_0.png)




```python
#center crop
center_crop = tfs.CenterCrop(100)(im)
center_crop
```




![png](output_4_0.png)




```python
#Random Flip horizontal
h_flip = tfs.RandomHorizontalFlip()(im)
h_flip
#vertical
v_flip = tfs.RandomVerticalFlip()(im)
v_flip
#Random Rotation
rot_im = tfs.RandomRotation(40)(im)#-40-40
rot_im
```




![png](output_5_0.png)




```python
#brightness
bright_im = tfs.ColorJitter(2)(im)
bright_im
```




![png](output_6_0.png)




```python
#contrast
contrast_im = tfs.ColorJitter(contrast=2)(im)
contrast_im
```




![png](output_7_0.png)




```python
#Hue
hue_im = tfs.ColorJitter(hue = 0.5)(im)
hue_im
```




![png](output_8_0.png)




```python
#Composite
im_aug = tfs.Compose([
    tfs.Resize(120),
    tfs.RandomHorizontalFlip(),
    tfs.RandomCrop(90),
    tfs.ColorJitter(brightness=0.5,contrast=0.5,hue = 0.5)
])
```


```python
import matplotlib.pyplot as plt
%matplotlib inline
nrows = 3
ncols = 3
figsize = (8,8)
_,figs = plt.subplots(nrows, ncols, figsize = figsize)
for i in range(nrows):
    for j in range (ncols):
        figs[i][j].imshow(im_aug(im))
        figs[i][j].axes.get_xaxis().set_visible(False)
        figs[i][j].axes.get_yaxis().set_visible(False)
plt.show()
```


![png](output_10_0.png)



```python
#***Training Model Test***
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.datasets import CIFAR10
import torchvision 
from torchvision import transforms as tfs
from utils import train, resnet


#data augmented
def train_tf(x):
    im_aug = tfs.Compose([
        tfs.Resize(120),
        tfs.RandomHorizontalFlip(),
        tfs.RandomCrop(96),
        tfs.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
        tfs.ToTensor(),
        tfs.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    x = im_aug(x)
    return x

def test_tf(x):
    im_aug = tfs.Compose([
        tfs.Resize(96),
        tfs.ToTensor(),
        tfs.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    x = im_aug(x)
    return x

train_set = CIFAR10('./data', train=True, transform=train_tf)
train_data = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
test_set = CIFAR10('./data', train=False, transform=test_tf)
test_data = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False)

net = resnet(3, 10)
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
train(net, train_data, test_data, 10, optimizer, criterion)

```


    ---------------------------------------------------------------------------

    IndexError                                Traceback (most recent call last)

    <ipython-input-68-03ee2ac4df45> in <module>
         41 optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
         42 criterion = nn.CrossEntropyLoss()
    ---> 43 train(net, train_data, test_data, 10, optimizer, criterion)
    

    ~/Documents/19Spring/Kaggle/utils.py in train(net, train_data, valid_data, num_epochs, optimizer, criterion)
         38             train_acc += get_acc(output, label)
         39 
    ---> 40         cur_time = datetime.now()
         41         h, remainder = divmod((cur_time - prev_time).seconds, 3600)
         42         m, s = divmod(remainder, 60)


    IndexError: invalid index of a 0-dim tensor. Use tensor.item() to convert a 0-dim tensor to a Python number



```python

```
