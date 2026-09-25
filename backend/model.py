import os
import urllib.request

import numpy as np
from scipy.ndimage import zoom
from scipy.special import logsumexp
import torch
import deepgaze_pytorch

DEVICE = "cpu"  # macOS, no CUDA
CENTERBIAS_PATH = "centerbias_mit1003.npy"
CENTERBIAS_URL = (
    "https://github.com/matthias-k/DeepGaze/releases/download/v1.0.0/centerbias_mit1003.npy"
)


class SaliencyModel:
    """DeepGaze IIE plus the center bias template, loaded once and reused."""

    def __init__(self, device=DEVICE, centerbias_path=CENTERBIAS_PATH):
        self.device = device

        if not os.path.exists(centerbias_path):
            urllib.request.urlretrieve(CENTERBIAS_URL, centerbias_path)

        self.centerbias_template = np.load(centerbias_path)
        self.model = deepgaze_pytorch.DeepGazeIIE(pretrained=True).to(device)
        self.model.eval()

    def _centerbias_for(self, height, width):
        centerbias = zoom(
            self.centerbias_template,
            (height / self.centerbias_template.shape[0],
             width / self.centerbias_template.shape[1]),
            order=0, mode="nearest",
        )
        return centerbias - logsumexp(centerbias)

    def predict(self, image):
        """Run inference on an RGB uint8 image and return a 2D log-density array."""
        height, width = image.shape[:2]

        image_tensor = torch.tensor([image.transpose(2, 0, 1)]).float().to(self.device)
        centerbias_tensor = (
            torch.tensor([self._centerbias_for(height, width)]).float().to(self.device)
        )

        with torch.no_grad():
            log_density = self.model(image_tensor, centerbias_tensor)

        return log_density.detach().cpu().numpy()[0, 0]
