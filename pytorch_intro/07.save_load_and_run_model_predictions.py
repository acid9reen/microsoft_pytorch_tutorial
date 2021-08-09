# %%
import torch
import torch.onnx as onnx
import torchvision.models as models


# %%
#Saving and loading model weights

# %%
# Save
model = models.vgg16(pretrained=True)
torch.save(model.state_dict(), r'data/model_weights.pth')

# %%
# Load
model = models.vgg16()
model.load_state_dict(torch.load(r'data/model_weights.pth'))
model.eval()

# %%
#Saving and loading models with shapes

# %%
# Save
torch.save(model, r'data/vgg_model.pth')

# %%
# Load
model = torch.load('data/vgg_model.pth')

# %%
# Exporting the model to ONNX
input_image = torch.zeros((1,3,224,224))
onnx.export(model, input_image, 'data/model.onnx')
