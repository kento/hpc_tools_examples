#!/bin/env python

import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
import horovod.torch as hvd

learning_rate = 1e-3
batch_size = 64
epochs = 1

class Net1(nn.Module):
        def __init__(self):
                super(Net1, self).__init__()
                self.flatten = nn.Flatten()
                self.fc1 = nn.Linear(28 * 28, 1000)
                self.fc2 = nn.Linear(1000, 10)

        def forward(self, x):
                x = self.flatten(x)
                x = self.fc1(x)
                x = F.relu(x)
                x = self.fc2(x)
                return x

# Initialize Horovod
hvd.init()
print(f"hvd.init() done:  {hvd.rank()} / {hvd.size()}")

train_dataset = datasets.MNIST(root="../data", train=True, download=True, transform=ToTensor())
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
if hvd.rank() == 0: print(f"sampler init done !")
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
if hvd.rank() == 0: print(f"loader init done !");

# Build model
#model = torchvision.models.resnet50()
model = Net1()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Add Horovod Distributed Optimizer
optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
if hvd.rank() == 0: print(f"optimizer init done !");

# Broadcast parameters from rank 0 to all other processes.
hvd.broadcast_parameters(model.state_dict(), root_rank=0)
if hvd.rank() == 0: print(f"bcast param done !");

for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{}]\tLoss: {}'.format(
               epoch, batch_idx * len(data), len(train_sampler), loss.item()))


