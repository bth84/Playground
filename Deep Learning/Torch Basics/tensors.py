import torch
import torch.nn as nn
import numpy as np

def describe(x):
    print("Type {}".format(x.type()))
    print("Shape/size: {}".format(x.shape))
    print("Values: \n{}".format(x))

print("___Creating a Tensor___\n")
print("torch.Tensor(2,3)")
describe(torch.Tensor(2,3))

print("torch.rand(2,3)")
describe(torch.rand(2,3))
print(torch.randn(2,3))
describe(torch.randn(2,3))

print("torch.zeros(2,3)")
describe(torch.zeros(2,3))
print("torch.ones(2,3)")
x = torch.ones(2,3)
describe(x)
print("x.fill_(5)")
x.fill_(5)
describe(x)

print("___from lists and Numpy___\n")
# noinspection PyArgumentList
x = torch.Tensor([[1,2,3],
                  [4,5,6]])
describe(x)

print("torch.from_numpy(npy)")
npy = np.random.rand(2,3)
describe(torch.from_numpy(npy))

