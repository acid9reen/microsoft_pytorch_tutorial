# %%
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from torchinfo import summary

from pytorchcv import load_mnist, train, plot_results
load_mnist(batch_size=128)


# %%
# Build neural network using sequential module
net = nn.Sequential(
    nn.Flatten(), 
    nn.Linear(784, 100),     # 784 inputs, 100 outputs
    nn.ReLU(),              # Activation Function
    nn.Linear(100, 10),      # 100 inputs, 10 outputs
    nn.LogSoftmax(dim=0)
)

summary(net,input_size=(1,28,28))

# %%
# Train our nn
hist = train(net, train_loader, test_loader, epochs=5)
plot_results(hist)

# %%
# Class-based network definitions

# %%
from torch.nn.functional import relu,log_softmax

class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.hidden = nn.Linear(784, 100)
        self.out = nn.Linear(100, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.hidden(x)
        x = relu(x)
        x = self.out(x)
        x = log_softmax(x, dim=0)
        return x

net = MyNet()

summary(net,input_size=(1,28,28))

# %%
hist = train(net, train_loader, test_loader, epochs=5)
plot_results(hist)
