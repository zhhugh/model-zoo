#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：model_zoo 
@File ：LeNet_torch.py
@Author ：zhouhan
@Date ：2022/3/1 8:15 下午 
'''

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch


class LeNet(nn.Module):
    def __init__(self, num_classes):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=(5, 5), padding=2)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(5, 5))
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    model = LeNet(num_classes=10)
    # x = np.random.uniform(size=(1, 1, 28, 28))
    # x = torch.tensor(x)
    #  = x.float()
    x = torch.rand(size=(1, 1, 28, 28))
    print(x.dtype)
    out = model(x)
    print(out)
