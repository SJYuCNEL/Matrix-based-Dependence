# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 21:01:42 2021

@author: Xi Yu
"""

from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import pandas as pd
import torch.nn as nn

class Syn_data(Dataset):
    def __init__(self, x, y, noise_type,noise_level):
        self.X = x
        self.y = y
        self.noise_type = noise_type
        self.noise_level = noise_level
         
    def __len__(self):
        return (len(self.X))
    
    def __getitem__(self, i):
        data = self.X[i]
        output = self.y[i]
        #output = output.reshape(-1,1)
        
        if self.noise_type=='Gaussian':
            noise = np.random.normal(0,self.noise_level)
        elif self.noise_type=='Laplace':
            noise = np.random.laplace(0, self.noise_level)
        elif self.noise_type=='Expoential':
            noise = self.noise_level*(1-np.random.exponential(1))
		elif self.noise_type=='None'
			nosie = 0
        output = np.array(output+noise).astype(np.float32)
        return (data, output)
    

def data_processing(rides):
    dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
    for each in dummy_fields:
        dummies = pd.get_dummies(rides[each], prefix=each, drop_first=False)
        rides = pd.concat([rides, dummies], axis=1)
    
    fields_to_drop = ['instant', 'dteday', 'season', 'weathersit', 
                      'weekday', 'atemp', 'mnth', 'workingday', 'hr']
    data = rides.drop(fields_to_drop, axis=1)
    quant_features = ['casual', 'registered', 'cnt', 'temp', 'hum', 'windspeed']
    # Store scalings in a dictionary so we can convert back later
    scaled_features = {}
    for each in quant_features:
        mean, std = data[each].mean(), data[each].std()
        scaled_features[each] = [mean, std]
        data.loc[:, each] = (data[each] - mean)/std
    target_fields = ['cnt', 'casual', 'registered']
    features, targets = np.array(data.drop(target_fields, axis=1)), np.array(data[target_fields[0]])
    data_process = np.array(data)
    return features, targets, data_process, scaled_features



