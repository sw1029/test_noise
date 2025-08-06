from base import Noise
import numpy as np

class GaussianNoise(Noise):
    @staticmethod
    def add_noise(src, mean=0, var=50) -> np.ndarray:
        sigma = var ** 0.5
        gaussian_noise = np.random.normal(mean, sigma, src.shape)
        noisy_image = src + gaussian_noise
        return np.clip(noisy_image, 0, 255).astype(np.uint8)