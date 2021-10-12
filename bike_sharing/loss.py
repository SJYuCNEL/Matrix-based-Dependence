# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 21:14:43 2021

@author: Xi Yu
"""

import torch
import torch.nn as nn

def pairwise_distances(x):
    #x should be two dimensional
    instances_norm = torch.sum(x**2,-1).reshape((-1,1))
    return -2*torch.mm(x,x.t()) + instances_norm + instances_norm.t()



def calculate_gram_mat(x, sigma):
    dist= pairwise_distances(x)
    return torch.exp(-dist /sigma)


def reyi_entropy(x,sigma):
    alpha = 1.001
    k = calculate_gram_mat(x,sigma)
    k = k/torch.trace(k) 
    eigv = torch.abs(torch.symeig(k, eigenvectors=True)[0])
    eig_pow = eigv**alpha
    entropy = (1/(1-alpha))*torch.log2(torch.sum(eig_pow))
    return entropy


def joint_entropy(x,y,s_x,s_y):
    alpha = 1.001
    x = calculate_gram_mat(x,s_x)
    y = calculate_gram_mat(y,s_y)
    k = torch.mul(x,y)
    k = k/torch.trace(k)
    eigv = torch.abs(torch.symeig(k, eigenvectors=True)[0])
    eig_pow =  eigv**alpha
    entropy = (1/(1-alpha))*torch.log2(torch.sum(eig_pow))

    return entropy


def calculate_MI(x,y,s_x,s_y):
    Hx = reyi_entropy(x,sigma=s_x)
    Hy = reyi_entropy(y,sigma=s_y)
    Hxy = joint_entropy(x,y,s_x,s_y)
    Ixy = Hx+Hy-Hxy
    normlize = Ixy/(torch.max(Hx,Hy)+1e-16)
    return normlize



def GaussianKernelMatrix(x, sigma):
    pairwise_distances_ = pairwise_distances(x)
    return torch.exp(-pairwise_distances_ /sigma)



def HSIC(x, y, s_x, s_y):

    m,_ = x.shape #batch size
    K = GaussianKernelMatrix(x,s_x)
    L = GaussianKernelMatrix(y,s_y)
    H = torch.eye(m) - 1.0/m * torch.ones((m,m))
    H = H.float().cuda()
    HSIC = torch.trace(torch.mm(L,torch.mm(H,torch.mm(K,H))))/((m-1)**2)
    return HSIC

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss(reduce=False)
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss
    
    
    
def loss_fn(inputs, outputs, targets, name):
    inputs_2d = inputs.view(inputs.shape[0], -1)
    error = targets - outputs
    #error = rmse(outputs, targets)
    if name == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, targets)
    if name == 'mse':
        criterion = nn.MSELoss()
        loss = criterion(outputs, targets)
    if name =='rmse':
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(outputs,targets) + 1e-6)
    if name =='MAE':
        criterion = torch.nn.L1Loss()
        loss = criterion(outputs, targets)
        
    if name == 'HSIC':
        loss = HSIC(inputs_2d, error, s_x=2, s_y=1)
    if name == 'ours':
        loss = calculate_MI(inputs_2d,error,s_x=2,s_y=1)
    if name == 'MEE':
        loss = reyi_entropy(error,sigma=1)
    if name =='bias':
        loss = targets - outputs
        loss = torch.mean(loss, 0)
    return loss