#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：model_zoo 
@File ：AlexNet_torch.py
@Author ：zhouhan
@Date ：2022/3/13 11:43 下午 
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 48, kernel_size=(11, 11), stride=(4, 4), padding=2)
        self.maxpool1 = nn.MaxPool2d((2, 2))
        self.conv2 = nn.Conv2d(48, 128, kernel_size=(5, 5), stride=(1, 1), padding=2)
        self.conv3 = nn.Conv2d(128, 192, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv4 = nn.Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv5 = nn.Conv2d(192, 128, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.fc1 = nn.Linear(4608, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 1000)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool1(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = F.relu(self.conv5(x))
        x = self.maxpool2(x)
        x = x.view(-1, 4608)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    # x = torch.rand(size=(1, 3, 224, 224))
    model = AlexNet()
    model.to(device)
    summary(model, (3, 224, 224), device='cuda')
    # out = model(x)
    # print(out)



