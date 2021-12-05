#!/bin/env python

# Directly from data
print ("--1--\n")
import torch
data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)
print (data)
print (x_data)

# From a NumPy array
import numpy as np
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print (data)
print (x_np)

# Mut and Add
print ("\n--2--\n")
print (x_np + x_np)
print (x_np * x_np)


