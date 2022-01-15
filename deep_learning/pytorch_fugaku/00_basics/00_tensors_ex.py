#!/bin/env python
import torch
A = torch.tensor([[1, 2, 3],[2, 3, 4], [3, 4, 5]])
B = torch.tensor([[5, 4, 3],[4, 3, 2], [3, 2, 1]])

print (A + B)
print (A * B)


