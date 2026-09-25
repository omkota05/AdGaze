import os
import urllib.request

import numpy as np
from PIL import Image
from scipy.ndimage import zoom
from scipy.special import logsumexp
import torch
import deepgaze_pytorch

DEVICE = 'cpu'  # macOS, no CUDA, keep it simple for this first test

# One-time download of the center bias file (small .npy, ships with the repo's release)
centerbias_path = 'centerbias_mit1003.npy'
if not os.path.exists(centerbias_path):
    url = 'https://github.com/matthias-k/DeepGaze/releases/download/v1.0.0/centerbias_mit1003.npy'
    urllib.request.urlretrieve(url, centerbias_path)

# Drop any jpg/png into this folder and point to it here
image = np.array(Image.open('test_image.jpg').convert('RGB'))

model = deepgaze_pytorch.DeepGazeIIE(pretrained=True).to(DEVICE)

centerbias_template = np.load(centerbias_path)
centerbias = zoom(
    centerbias_template,
    (image.shape[0] / centerbias_template.shape[0],
     image.shape[1] / centerbias_template.shape[1]),
    order=0, mode='nearest'
)
centerbias -= logsumexp(centerbias)

image_tensor = torch.tensor([image.transpose(2, 0, 1)]).float().to(DEVICE)
centerbias_tensor = torch.tensor([centerbias]).float().to(DEVICE)

log_density_prediction = model(image_tensor, centerbias_tensor)

print("Output shape:", log_density_prediction.shape)
print("Success! Saliency map generated.")