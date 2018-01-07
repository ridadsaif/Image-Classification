# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 16:27:48 2018

@author: Hp
"""

import numpy as np
import csv
import torch
import torch.nn as nn

from torch.autograd import Variable
import torchvision.transforms as transforms
import pdb
import scipy.misc
import pandas as pd
import matplotlib.pyplot as plt

ljk = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 24, 25, 27, 28, 30, 32, 35, 36, 40, 42, 45, 48, 49, 54, 56, 63, 64, 72, 81]
dict_transform = dict((ljk[i],i) for i in range(len(ljk)))

def data_load(x_path,y_path):
    x=np.loadtxt(x_path,delimiter=",")
    y=np.loadtxt(y_path,delimiter=",")
    x=x.reshape(-1,64,64)
    y=y.reshape(-1,1)
    print("Data Loaded")
    return x,y


def data_load(x_path,y_path):
    x=np.loadtxt(x_path,delimiter=",")
    y=np.loadtxt(y_path,delimiter=",")
    x=x.reshape(-1,64,64)
    y=y.reshape(-1,1)
    print("Data Loaded")
    return x,y


def cross_validation(X_train,Y_train,split):
    examples_train=int(split*np.shape(X_train)[0])
    examples_test=np.shape(X_train)[0]-examples_train
    train_set_x=np.zeros((examples_train,64,64))
    train_set_y=np.zeros((examples_train,1))
    test_set_x=np.zeros((examples_test,64,64))
    test_set_y=np.zeros((examples_test,1))
    
    for i in range(examples_train):
        train_set_x[i]=X_train[i]
        train_set_y[i]=Y_train[i]
        
    for j in range(examples_test):
        test_set_x[j]=X_train[j+examples_train]
        test_set_y[j]=Y_train[j+examples_train]
    return train_set_x,train_set_y,test_set_x,test_set_y

num_epochs=13
batch_size=120
#np.shape(X_train)[0]

learning_rate=0.01
# print(train_set_y[0:20,0])

train_set_x=train_set_x.astype(np.float32).reshape(-1,1,64,64)
test_set_x=test_set_x.astype(np.float32).reshape(-1,1,64,64)

kaggle_set_x=kaggle_set_x.astype(np.float32).reshape(-1,1,64,64)

features_train=torch.from_numpy(train_set_x)
features_test=torch.from_numpy(test_set_x)
features_kaggle=torch.from_numpy(kaggle_set_x)

features_train=features_train.contiguous()
features_test=features_test.contiguous()
features_kaggle=features_kaggle.contiguous()




labels_train=train_set_y.astype(np.float32)
for i in range(np.shape(labels_train)[0]):
    labels_train[i,0]=dict_transform[labels_train[i,0]]
#print(labels[0:50])
labels_train=labels_train.reshape(np.shape(labels_train)[0])
labels_train=torch.Tensor(labels_train)


labels_test=test_set_y.astype(np.float32)
for i in range(np.shape(labels_test)[0]):
    labels_test[i,0]=dict_transform[labels_test[i,0]]
#print(labels[0:50])
labels_test=labels_test.reshape(np.shape(labels_test)[0])
labels_test=torch.Tensor(labels_test)



train=torch.utils.data.TensorDataset(features_train,labels_train)
test=torch.utils.data.TensorDataset(features_test,labels_test)


train_set_dataloader=torch.utils.data.DataLoader(dataset=train,batch_size=batch_size,shuffle=False)
test_set_dataloader=torch.utils.data.DataLoader(dataset=test,batch_size=batch_size,shuffle=False)
kaggle_loader=torch.utils.data.DataLoader(dataset=features_kaggle,batch_size=batch_size,shuffle=False)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            #nn.Dropout(p=0.2),
            nn.MaxPool2d(2))
             
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=5, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            #nn.Dropout(p=0.2),
            nn.MaxPool2d(2))
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 1024, kernel_size=5, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            #nn.Dropout(p=0.2),
            nn.MaxPool2d(2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            #nn.Dropout(p=0.2),
            nn.MaxPool2d(2))
        
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            ##nn.Dropout(p=0.2),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(128,40)
        
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out=self.layer3(out)
        out=self.layer4(out)
        out=self.layer5(out)
        out=self.layer6(out)
        out = out.view(out.size(0), -1)
        #print(out.size())
        out = self.fc(out)
        return out
    

cnn = CNN()
cnn.cuda()
all_loss=[]

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
# Train the Model
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_set_dataloader):
        
        images = Variable(images.cuda())
        labels = Variable(labels.cuda())
        #print(images.size())
        
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        #pdb.set_trace()
        
        
        #images=images.unsqueeze(0)
    
        outputs = cnn(images)
        #pdb.set_trace()
        #print(labels.long())
        loss = criterion(outputs, labels.long())
                         
        loss.backward()
        optimizer.step()
        
        
        if (i+1) % 100 == 0:
            all_loss.append(loss.data[0])
            print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' 
                   %(epoch+1, num_epochs, i+1, len(train)//batch_size, loss.data[0]))
            

# Test the Model
cnn.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
correct = 0
total = 0
for images, labels in test_set_dataloader:
    images = Variable(images.cuda())
    outputs = cnn(images.cuda())
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    #print(type(predicted))
    correct += (predicted == labels.long().cuda()).sum()

print('Test Accuracy of the model on the 15000 test images: %d %%' % (100 * correct / total))


###saving the predictions in kaggle
#f=open('results_kaggle.csv','w')
#print('Id,Label',file=f)
output = []
counter=1
for images in kaggle_loader:
    images=Variable(images.cuda())
    outputs=cnn(images)
    _,predicted=torch.max(outputs.data,1)
    #print(predicted)
   # print(counter,',',predicted,file=f)
    predictions=predicted.cpu().numpy()
    output.extend(predictions)
    
    #print(predictions.shape)
    #counter+=1
df = pd.DataFrame(list(zip(range(1,len(output)+1),output)), columns=['Id','RawLabels'])
#f.close()