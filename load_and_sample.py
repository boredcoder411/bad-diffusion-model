import torch
from diffusion_models import DiffusionModel
from unet import UNet

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = UNet()
model.load_state_dict(torch.load('model.pth'))
model.eval()
diffusion_model = DiffusionModel(1000, model, device)
samples = diffusion_model.sampling(n_samples=1, use_tqdm=False)
