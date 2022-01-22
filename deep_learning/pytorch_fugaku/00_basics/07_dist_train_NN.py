#!/bin/env python

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch import nn
from torch.nn import functional as F
import horovod.torch as hvd
import time

hvd.init()

learning_rate = 1e-3
batch_size = 64
epochs = 1
torch.set_num_threads(2)

training_data = datasets.MNIST(
        root="../data",
        train=True,
        download=True,
        transform=ToTensor()
    )

test_data = datasets.MNIST(
        root="../data",
        train=False,
        download=True,
        transform=ToTensor()
    )

train_sampler = torch.utils.data.distributed.DistributedSampler(training_data, num_replicas=hvd.size(), rank=hvd.rank())
test_sampler  = torch.utils.data.distributed.DistributedSampler(test_data, num_replicas=hvd.size(), rank=hvd.rank()) 

train_dataloader = DataLoader(dataset=training_data, batch_size=batch_size, sampler=train_sampler)
test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, sampler=test_sampler)

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

model = Net1()

# Training

# Initialize the loss function
loss_fn = nn.CrossEntropyLoss()

# We initialize the optimizer by registering the model's parameters that need to be trained, and passing in the learning rate hyperparameter
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate * hvd.size())
optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters()) 

hvd.broadcast_parameters(model.state_dict(), root_rank=0) 

# evaluates the model's performance against our test data.
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            if hvd.rank() == 0: print(f"loss: {loss:>7f}  [{current:>5d}/{len(train_sampler):>5d}]", flush=True)


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.sampler)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    if hvd.rank() == 0: print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n", flush=True)

s = time.time()    
for t in range(epochs):
    if hvd.rank() == 0: print(f"Epoch {t+1}\n-------------------------------", flush=True)
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
e = time.time()    
if hvd.rank() == 0: print(f"Training Done! training time = {e-s:>2f}")


# Saving model
torch.save(model.state_dict(), "Net1_weights.pth")
if hvd.rank() == 0: print("Saving Done!")



                        
