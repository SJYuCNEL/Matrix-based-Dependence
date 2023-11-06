# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 22:34:57 2020

@author: yuxi1
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from torchvision import datasets, transforms
from models import covnet

# Device configuration
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--num_epoch',default=100, 
                    type=int, help='num_epoch')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch







# Hyper parameters

num_epochs = 20
num_classes = 10
batch_size = 100
learning_rate = 0.001


# Fashion MNIST dataset

train_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5,), (0.5,))
                   ])),

    batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('../data', train=False, 
                   transform=transforms.Compose([
                   transforms.ToTensor(),
                   transforms.Normalize((0.5,), (0.5,))
    ])),
    batch_size=batch_size, shuffle=True)

# test_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=False, transform=transforms.Compose([
#         transforms.RandomRotation(degrees=(45, -45)),
#         transforms.ToTensor(),
        
#         transforms.Normalize((0.1307,), (0.3081,))
#     ])),
#     batch_size=batch_size, shuffle=True)

#%%
images, labels = next(iter(test_loader))
plt.imshow(images[0].view(28, 28), cmap="gray")
plt.show()

#%%
# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, num_classes)

        

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out
    
net = ConvNet(num_classes=10)
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

# Loss and optimizer


optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
def pairwise_distances(x):
    #x should be two dimensional
    instances_norm = torch.sum(x**2,-1).reshape((-1,1))
    return -2*torch.mm(x,x.t()) + instances_norm + instances_norm.t()



def calculate_gram_mat(x, sigma):
    dist= pairwise_distances(x)
    #dist = dist/torch.max(dist)
    return torch.exp(-dist /sigma)


def reyi_entropy(x,sigma):
    alpha = 1
    k = calculate_gram_mat(x,sigma)
    k = k/torch.trace(k) 
    eigv = torch.abs(torch.symeig(k, eigenvectors=True)[0])
    eig_pow = eigv**alpha
    entropy = (1/(1-alpha))*torch.log2(torch.sum(eig_pow))
    return entropy


def joint_entropy(x,y,s_x,s_y):
    alpha = 1
    x = calculate_gram_mat(x,s_x)
    y = calculate_gram_mat(y,s_y)
    k = torch.mul(x,y)
    k = k/torch.trace(k)
    eigv = torch.abs(torch.symeig(k, eigenvectors=True)[0])
    eig_pow =  eigv**alpha
    entropy = (1/(1-alpha))*torch.log2(torch.sum(eig_pow))

    return entropy


def calculate_MI(x,y,s_x,s_y):
    return (reyi_entropy(x,sigma=s_x)+reyi_entropy(y,sigma=s_y)-\
            joint_entropy(x,y,s_x,s_y))/(joint_entropy(x,y,s_x,s_y)+1e-16)



def GaussianKernelMatrix(x, sigma=1):

    pairwise_distances_ = pairwise_distances(x)
    return torch.exp(-pairwise_distances_ /sigma)



def HSIC(x, y, s_x=1, s_y=1):

    m,_ = x.shape #batch size
    K = GaussianKernelMatrix(x,s_x)
    L = GaussianKernelMatrix(y,s_y)
    H = torch.eye(m) - 1.0/m * torch.ones((m,m))
    H = H.float().cuda()
    HSIC = torch.trace(torch.mm(L,torch.mm(H,torch.mm(K,H))))/((m-1)**2)
    return HSIC


def loss_fn(inputs, outputs, targets, name):
    inputs_2d = inputs.view(batch_size, -1)
    error = F.softmax(outputs,dim=1) - F.one_hot(targets,10)
    if name == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, targets)
    if name == 'mse':
        criterion = nn.MSELoss()
        loss = criterion(F.softmax(outputs,dim=1), F.one_hot(targets,10))
    if name == 'HSIC':
        loss = HSIC(inputs_2d, error, s_x=1, s_y=1)
    if name == 'MEE':
        loss = calculate_MI(inputs_2d,error,s_x=1,s_y=1)
    return loss

#%%

# Train the model

# Training
def train(epoch):
    #print('\nEpoch: %d' % epoch)
    print('\nEpoch [{}/{}]'.format(epoch+1, args.num_epoch))
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        
        loss = loss_fn(inputs, outputs, targets, 'MEE')
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    print ('Step [{}/{}], Loss: {:.4f}, Acc: {}% [{}/{}])' 
               .format(batch_idx, 
                       len(train_loader), 
                       train_loss/(batch_idx+1),
                       100.*correct/total, correct, total))
    return train_loss/(batch_idx+1),100.*correct/total
       


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = loss_fn(inputs, outputs, targets, 'mse')

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        print ('Step [{}/{}], Loss: {:.4f}, Acc: {}% [{}/{}])' 
                   .format(batch_idx, 
                           len(test_loader), 
                           test_loss/(batch_idx+1),
                           100.*correct/total, correct, total))
        

    

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc
    return test_loss/(batch_idx+1),100.*correct/total

#%%
import time
train_MI = []
train_Accuracy = []
test_mse_Loss = []
test_Acc = []
for epoch in range(args.num_epoch):
    start_time = time.time()
    train_mi,train_accuracy = train(epoch)
    test_mse_loss,test_accuracy = test(epoch)
    
    print(time.time() - start_time)
    
    train_MI.append(train_mi)
    train_Accuracy.append(train_accuracy)
    test_mse_Loss.append(test_mse_loss)
    test_Acc.append(test_accuracy)
    #scheduler.step()
    