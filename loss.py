# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 16:13:45 2020

@author: Shujian Yu, Xi Yu
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from MI import calculate_MI, renyi_entropy

def pairwise_distances(x):
    #x should be two dimensional
    instances_norm = torch.sum(x**2,-1).reshape((-1,1))
    return -2*torch.mm(x,x.t()) + instances_norm + instances_norm.t()

def GaussianKernelMatrix(x, sigma=1):
    pairwise_distances_ = pairwise_distances(x)
    return torch.exp(-pairwise_distances_ /sigma)


def HSIC(x, y, s_x, s_y):
    
    """ calculate HSIC from https://github.com/danielgreenfeld3/XIC"""
    m,_ = x.shape #batch size
    K = GaussianKernelMatrix(x,s_x)
    L = GaussianKernelMatrix(y,s_y)
    H = torch.eye(m) - 1.0/m * torch.ones((m,m))
    H = H.float().cuda()
    HSIC = torch.trace(torch.mm(L,torch.mm(H,torch.mm(K,H))))/((m-1)**2)
    return HSIC

def loss_fn(inputs, outputs, targets, name):
    """ loss function for different method"""
    inputs_2d = inputs.view(inputs.shape[0], -1) #input space
    error = F.softmax(outputs,dim=1) - F.one_hot(targets,10) #error space
    if name == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, targets)
    if name == 'mse':
        criterion = nn.MSELoss()
        loss = criterion(F.softmax(outputs,dim=1), F.one_hot(targets,10))
    if name == 'HSIC':
        loss = HSIC(inputs_2d, error, s_x=1, s_y=1)
    if name == 'MBD':
        loss = calculate_MI(inputs_2d,error,s_x=1,s_y=20)
    if name == 'MEE':
        loss = renyi_entropy(error,sigma=1)
    if name =='bias':
        loss = F.one_hot(targets,10)-F.softmax(outputs,dim=1)
        loss = torch.mean(loss, 0)
    return loss


