from base import Noise
import numpy as np  

class PoissonNoise(Noise):
    def __init__(self,src):
        self.src = src
        self.rows, self.cols, self.channels = src.shape
    def add_noise(self) -> np.ndarray:
        noisy_image = np.copy(self.src)
        for i in range(self.src.shape[2]):  # 채널별로 독립적으로 적용
            noisy_image[:, :, i] = np.random.poisson(self.src[:, :, i]).astype(np.uint8)
        return noisy_image