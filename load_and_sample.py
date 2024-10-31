import torch
from diffusion_models import DiffusionModel
from unet import UNet
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load model
model = UNet()
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Create diffusion model
diffusion_model = DiffusionModel(1000, model, device)

# Sampling
samples = diffusion_model.sampling(n_samples=5)
samples = samples.cpu().numpy()
samples = samples.transpose(0, 2, 3, 1)

# Plot the samples
fig, axes = plt.subplots(1, 5, figsize=(20, 5))
for i, ax in enumerate(axes):
    ax.imshow(samples[i])
    ax.axis('off')
    
plt.savefig('samples.png')