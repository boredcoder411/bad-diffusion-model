import torch
from diffusion_models import DiffusionModel
from unet import UNet
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = UNet()
model.load_state_dict(torch.load('model.pth'))
model.eval()
diffusion_model = DiffusionModel(1000, model, device)
samples = diffusion_model.sampling(n_samples=1, use_tqdm=False)

plt.imshow(samples[0].squeeze(0).clip(0, 1).data.cpu().numpy(), cmap='gray')
plt.axis('off')
plt.savefig('Imgs/sample.png')