# %%
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import Lambda, ToTensor
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# %%
# Loading a dataset
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

# %%
# Iterating and Visualizing the Dataset
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

fig = plt.figure(figsize=(8, 8))
cols, rows = 3, 3

for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    fig.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="bone")

plt.show()

# %%
# Creating a Custom Dataset for your files

# %%
# A custom Dataset class must implement three functions: 
# __init__, __len__, and __getitem__

# %%
import os
import pandas as pd
import torchvision.io as tvio


# %%
class CustomImageDataset(Dataset):
    def __init__(
        self, 
        annotations_file, 
        img_dir, 
        transform=None, 
        target_transform=None
    ) -> None:
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = tvio.read_image(img_path)
        label = self.img_labels.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        sample = {"image": image, "label": label}

        return sample


# %%
# Preparing your data for training with DataLoaders
from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# %%
# Iterate through the DataLoader
# Display image and label.
train_features, train_labels = next(iter(train_dataloader))

print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")

img = train_features[0].squeeze()
label = train_labels[0]

plt.imshow(img, cmap="viridis")
plt.show()

print(f"Label: {labels_map[label.item()]}")

# %%
# Conclusion:
# DataSet is designed for retrieval of individual data items 
# while a DataLoader is designed to work with batches of data
