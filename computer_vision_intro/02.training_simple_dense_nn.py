# %%
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt

import pytorchcv

pytorchcv.load_mnist()

# %%
net = nn.Sequential(
    nn.Flatten(), nn.Linear(28 * 28, 10), nn.LogSoftmax()  # 784 inputs, 10 outputs
)

# %%
print("Digit to be predicted: ", data_train[0][1])
torch.exp(net(data_train[0][0]))

# %%
# Train nn

# %%
train_loader = torch.utils.data.DataLoader(data_train, batch_size=64)
test_loader = torch.utils.data.DataLoader(data_test, batch_size=64)

# %%
def train_epoch(net, dataloader, lr=0.01, optimizer=None, loss_fn=nn.NLLLoss()):
    optimizer = optimizer or torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    total_loss, acc, count = 0, 0, 0

    for features, labels in dataloader:
        optimizer.zero_grad()
        out = net(features)
        loss = loss_fn(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss
        __, predicted = torch.max(out, 1)
        acc += (predicted == labels).sum()
        count += len(labels)

    return total_loss.item() / count, acc.item() / count


train_epoch(net, train_loader)

# %%
def validate(net, dataloader, loss_fn=nn.NLLLoss()):
    net.eval()
    count, acc, loss = 0, 0, 0

    with torch.no_grad():
        for features, labels in dataloader:
            out = net(features)
            loss += loss_fn(out, labels)
            pred = torch.max(out, 1)[1]
            acc += (pred == labels).sum()
            count += len(labels)

    return loss.item() / count, acc.item() / count


validate(net, test_loader)

# %%
def train(
    net,
    train_loader,
    test_loader,
    optimizer=None,
    lr=0.01,
    epochs=10,
    loss_fn=nn.NLLLoss(),
):
    optimizer = optimizer or torch.optim.Adam(net.parameters(), lr=lr)
    res = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for ep in range(epochs):
        tl, ta = train_epoch(
            net, train_loader, optimizer=optimizer, lr=lr, loss_fn=loss_fn
        )

        vl, va = validate(net, test_loader, loss_fn=loss_fn)

        print(
            f"Epoch {ep:2}, Train acc={ta:.3f}, "
            f"Val acc={va:.3f}, Train loss={tl:.3f}, Val loss={vl:.3f}"
        )

        res["train_loss"].append(tl)
        res["train_acc"].append(ta)
        res["val_loss"].append(vl)
        res["val_acc"].append(va)

    return res


# Re-initialize the network to start from scratch
net = nn.Sequential(
    nn.Flatten(), nn.Linear(784, 10), nn.LogSoftmax()  # 784 inputs, 10 outputs
)

hist = train(net, train_loader, test_loader, epochs=5)

# %%
plt.figure(figsize=(15,5))
plt.subplot(121)
plt.plot(hist['train_acc'], label='Training acc')
plt.plot(hist['val_acc'], label='Validation acc')
plt.legend()

plt.subplot(122)
plt.plot(hist['train_loss'], label='Training loss')
plt.plot(hist['val_loss'], label='Validation loss')
plt.legend()

plt.show()

# %%
# Visualizing network weights
weight_tensor = next(net.parameters())
fig, axes = plt.subplots(1, 10, figsize=(15,4))

for i, (ax, x)  in enumerate(zip(axes, weight_tensor)):
    ax.imshow(x.view(28,28).detach())
