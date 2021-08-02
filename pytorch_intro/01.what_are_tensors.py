# %%
import torch
import numpy as np


# %%
# Initializing a Tensor

# %%
# Directly from data
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

print(f"Tensor from data: {x_data}")

# %%
# From numpy ndarray
np_array = np.array(data)
x_np = torch.tensor(np_array)

print(f"Tensor from numpy: {x_np}")

# %%
# With random or constant values
shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random tensor: {rand_tensor}")
print(f"Ones tensor: {ones_tensor}")
print(f"Zeros tensor: {zeros_tensor}")

# %%
# Attributes of tensor
tensor = torch.rand((3, 4,))
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Tensor device: {tensor.device}")

# %%
# Operations on tensor

# %%
# Move tensor to gpu if available
if torch.cuda.is_available():
    tensor = tensor.to("cuda")

# %%
# Standard numpy-like indexing and slicing
tensor = torch.ones(4, 4)
print(f"Tensor: {tensor}")
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last_column: {tensor[..., -1]}")

tensor[:, 1] = 0
print(f"Modified tensor: {tensor}")

# %%
# Joining tensors
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(f"Concatenated tensor: {t1}")

# %%
# Arithmetic operations
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
print(f"Matrix multiplication: {y1, y2}")

z1 = tensor * tensor
z2 = tensor.mul(tensor)
print(f"Element-wise product: {z1, z2}")

# %%
# Single-element tensors
agg = tensor.sum() # This is tensor!

# Converting tensor to python numerical value
agg_item = agg.item()
print(f"tensor.sum() result: {type(agg)}")
print(f"agg.item() result: {type(agg_item)}")

# %%
# In-place operations
# Better not to use (Save some memory, but messy and have some derivateves limitations)
# Ends with _
tensor.add_(5)
print(f"In-place modified tensor: {tensor}")

# %%
# Bridge with numpy

# %%
# Tensor to numpy array
t = torch.ones(5)
n = t.numpy()
print(f"Original tensor: {t}")
print(f"Converted tensor: {n}")

# Note: converted object references to original!
t.add_(1)
print(f"Tensor: {t}")
print(f"Numpy array: {n}")

# %%
# Numpy array to tensor
n = np.ones(5)
t = torch.from_numpy(n)

print(f"Original array: {n}")
print(f"Converted array: {t}")

# Note: converted object references to original!
n += 1
print(f"Numpy array: {n}")
print(f"Tensor: {t}")
