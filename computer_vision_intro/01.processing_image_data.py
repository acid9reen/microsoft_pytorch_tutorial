# %%
import torch
from torch.utils import data
import torchvision
import matplotlib.pyplot as plt
import numpy as np


# %%
# Load MNIST dataset
from torchvision.transforms import ToTensor

data_train = torchvision.datasets.MNIST(
    r"./data",
    download=True,
    train=True,
    transform=ToTensor()
)

data_test = torchvision.datasets.MNIST(
    r"./data",
    download=True,
    train=False,
    transform=ToTensor()
)

# %%
# Visualize the digits dataset
fig, axes = plt.subplots(1, 7)
for ind, (ax, data) in enumerate(zip(axes, data_train)):
    ax.imshow(data[0].view(28, 28))
    ax.set_title(data[1])
    ax.axis("off")

# %%
# Dataset structure
print('Training samples:',len(data_train))
print('Test samples:',len(data_test))

print('Tensor size:',data_train[0][0].size())
print('First 10 digits are:',
      [data_train[i][1] for i in range(10)])

# %%
print('Min intensity value: ', data_train[0][0].min().item())
print('Max intensity value: ', data_train[0][0].max().item())
