#!/bin/env python

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

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

train_dataloader = DataLoader(dataset=training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(dataset=test_data, batch_size=64, shuffle=True)

# Display image and label.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.savefig("02_dataloaders.png")
print(f"Label: {label}")
