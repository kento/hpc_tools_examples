#!/bin/env python

import os
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
       
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

training_data = datasets.MNIST(root="../data", train=True, download=True, transform=ToTensor())
test_data = datasets.MNIST(root="../data", train=False, download=True,transform=ToTensor())

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

model = Net1()

learning_rate = 1e-3
batch_size = 64
epochs = 1

# Initialize the loss function
loss_fn = nn.CrossEntropyLoss()

# We initialize the optimizer by registering the model's parameters that need to be trained, and passing in the learning rate hyperparameter
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

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
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")



                        
