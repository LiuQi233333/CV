#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 21:02:26 2019

@author: pxb
"""

#coding = utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from torch import optim

#parameters
epochs = 2
batch_size = 100
lr = 0.01
download_mnist = False

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(    #(1,28,28)
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),   #-->（16，28，28）
            nn.ReLU(),#-->（16，28，28）
            nn.MaxPool2d(kernel_size=2),#-->（16，14，14）
        )
        self.conv2 = nn.Sequential(#（16，14，14）
            nn.Conv2d(16,32,5,1,2),#-->（32，14，14）
            nn.ReLU(),#-->（32，14，14）
            nn.MaxPool2d(2)#-->（32，7，7）

        )
        self.out = nn.Linear(32*7*7,10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)   #-->(batch,32,7,7)
        x = x.view(x.size(0),-1) #-->(batch,32*7*7)
        output = self.out(x)
        return output

def main():
    
    train_data = torchvision.datasets.MNIST(
        root='./minst_data',
        train=True,
        transform = torchvision.transforms.ToTensor(),
        download=download_mnist
    )
    
    train_loader = Data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_data = torchvision.datasets.MNIST(
        root='./minst_data',
        train=False,
    )
    test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1), volatile=True).type(torch.FloatTensor)[0:2000]/255.
    test_y = test_data.test_labels[0:2000]

    cnn = CNN()
    #print(cnn)
    
    opimizer = optim.Adam(cnn.parameters(),lr=lr)
    
    loss_func = nn.CrossEntropyLoss()
    
    #train and test
    accuracyList = []
    lossList = []
    
    for epoch in range(epochs):
        for step,(x,y) in enumerate(train_loader):
            b_x = Variable(x)
            b_y = Variable(y)            
    
            out = cnn(b_x)
            loss = loss_func(out,b_y)
            opimizer.zero_grad()
            loss.backward()
            opimizer.step()
    
            if step % 50 == 0:
                test_out = cnn(test_x)
                pred_y = torch.max(test_out,1)[1].data.squeeze()
                count = 0
                for i in range(test_y.shape[0]):
                    if(pred_y[i] == test_y[i]):
                        count += 1
                accuracy = count / test_y.shape[0]
                accuracyList.append(accuracy)
                lossList.append(loss.item())
                print('Epoch:',epoch,'|train loss:'+str(loss.item())+' accuracy is:'+str(accuracy))
                
    for i in range(test_y.shape[0]):
        if(pred_y[i] != test_y[i]):
            error_detail = {'label': test_y[i], 'error': pred_y[i]}
            print(error_detail)

    
    plt.title('accuracy')
    plt.plot(accuracyList)
    plt.show()
    plt.title('loss')
    plt.plot(lossList)
    plt.show()

if __name__ == '__main__':
    main()