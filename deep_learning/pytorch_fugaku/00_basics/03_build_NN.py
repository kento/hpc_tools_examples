#!/bin/env python

import os
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class Net1Seq(nn.Module):
        def __init__(self):
                super(Net1Seq, self).__init__()
                self.seq = nn.Sequential(
                        nn.Linear(28*28, 1000),
                        nn.ReLU(),
                        nn.Linear(1000, 10)
                )

        def forward(self, x):
                x = self.seq(x)
                return nn.Softmax(dim=0)(x)
            
model1Seq = Net1Seq()
print(model1Seq)
print(model1Seq(x))
                        
                        
