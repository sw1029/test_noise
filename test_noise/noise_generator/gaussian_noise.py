from base import Noise
import numpy as np

class gaussianNoise(Noise):
    def __init__(self,src, mean=0, var=50):
        self.src = src
        self.rows, self.cols, self.channels = src.shape
        self.mean = mean
        self.var = var
    def add_noise(self) -> np.ndarray:
        sigma = self.var ** 0.5
        gaussian_noise = np.random.normal(self.mean, sigma, self.src.shape)
        noisy_image = self.src + gaussian_noise
        return np.clip(noisy_image, 0, 255).astype(np.uint8)