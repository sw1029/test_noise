from base import Noise
import numpy as np

class saltPepperNoise(Noise):
    def __init__(self, src, s_vs_p = 0.5, amount = 0.02):
        self.src = src
        self.rows, self.cols, self.channels = src.shape
        self.s_vs_p = s_vs_p
        self.amount = amount
    def add_noise(self) -> np.ndarray:
        noisy_image = np.copy(self.src)
        
        # Salt 노이즈 (흰색) 추가
        num_salt = np.ceil(self.amount * self.src.size * self.s_vs_p / self.src.shape[2])  # 채널 수로 나누어 픽셀 수로 맞춤
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in self.src.shape[:2]]
        noisy_image[coords[0], coords[1], :] = 255  # 해당 좌표에 흰색 노이즈 적용

        # Pepper 노이즈 (검은색) 추가
        num_pepper = np.ceil(self.amount * self.src.size * (1.0 - self.s_vs_p) / self.src.shape[2])
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in self.src.shape[:2]]
        noisy_image[coords[0], coords[1], :] = 0  # 해당 좌표에 검은색 노이즈 적용

        return noisy_image