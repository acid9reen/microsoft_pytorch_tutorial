# %%
import torch


# %%
x = torch.ones(5) # input tensor
y = torch.zeros(3) # expected output
w = torch.rand(5, 3, requires_grad=True)
b = torch.rand(3, requires_grad=True)
z = x @ w + b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
print(f"{w=}")
print(f"{b=}")

print(f"{z=}")
print(f"{y=}")
print(f"Loss: {loss}")

# Note: You can set the value of requires_grad when creating a tensor, 
# or later by using x.requires_grad_(True) method.

# %%
print(f'Gradient function for z = {z.grad_fn}')
print(f'Gradient function for loss = {loss.grad_fn}')

# %%
loss.backward()
print(f"{w=}")
print(f"{b=}")

# %%
#Disabling gradient tracking
z = x @ w + b
print(z.requires_grad)

with torch.no_grad():
    z = x @ w + b

print(z.requires_grad)

# %%
# Another way to achieve the same result:
z = x @ w + b
z_det = z.detach()

print(z_det.requires_grad)
