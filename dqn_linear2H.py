# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 13:23:15 2020

@author: mjkam
"""
import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self , hidden_layer_size,input_size,output_size,dropout=0.0):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.fc1 = nn.Linear(input_size,hidden_layer_size[0])
        self.fc2 = nn.Linear(hidden_layer_size[0],hidden_layer_size[1])
        self.fcN = nn.Linear(hidden_layer_size[1],output_size)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        h = torch.relu(self.fc1(x))
        h = self.dropout(h)
        h = torch.relu(self.fc2(h))
        h = self.dropout(h)
        #h = torch.relu(self.fc4(h))
        h = self.fcN(h)
        return h
