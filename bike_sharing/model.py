# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 21:20:43 2021

@author: Xi Yu
"""
import torch
import torch.nn as nn

class LinearRegression(nn.Module):
    def __init__(self, input_shape, num_classes=1):
        super().__init__()
        self.fc1 = nn.Linear(input_shape, 100)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(100, 100)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(100, 10)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(10, num_classes)
    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.relu1(x1)
        x3 = self.fc2(x2)
        x4 = self.relu2(x3)
        x5 = self.fc3(x4)
        x6 = self.relu3(x5)
        x_out = self.fc4(x6)
        
        return x_out