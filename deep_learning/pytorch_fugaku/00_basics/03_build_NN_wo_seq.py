#!/bin/env python

import os
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class Net1(nn.Module):
        def __init__(self):
                super(Net1, self).__init__()
                self.fc1 = nn.Linear(28 * 28, 1000)
                self.fc2 = nn.Linear(1000, 10)

        def forward(self, x):
                x = self.fc1(x)
                x = F.relu(x)
                x = self.fc2(x)
                return nn.Softmax(dim=0)(x)
            
model1 = Net1()
print(model1)
x = torch.rand(28 * 28)
sm = model1(x)
print(sm)
print(sm.argmax())
                        
                        
