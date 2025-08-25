from .base import Noise
import numpy as np  

class PoissonNoise(Noise):
    @staticmethod
    def add_noise(src, factor = 0.1) -> np.ndarray:
        noisy_image = np.copy(src)
        for i in range(src.shape[2]):  # 채널별로 독립적으로 적용
            noisy_image[:, :, i] = np.random.poisson(src[:, :, i]).astype(np.uint8)
        noisy_image = src * (1 - factor) + noisy_image * factor
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        return noisy_image