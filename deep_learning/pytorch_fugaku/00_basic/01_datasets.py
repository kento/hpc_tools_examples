#!/bin/env python

import torch
from torch.utils.data import Dataset
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
