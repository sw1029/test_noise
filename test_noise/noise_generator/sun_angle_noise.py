from .base import Noise
import numpy as np
from ..utils import *

'''
기존 구현 코드의 absolute correction의 역연산을 취한 부분을 그대로 사용한다.
'''

class SunAngleNoise(Noise):
    @staticmethod
    def add_noise(src, angle=45, intensity=0.5, gamma=1.0):
        """
        태양 고도각에 따른 노이즈를 생성합니다.

        Parameters:
        src (numpy.ndarray): 원본 이미지
        angle (float): 태양의 고도각 (0 ~ 90도)
        intensity (float): 조명의 강도 (0 ~ 1 사이의 값)
        gamma (float): 감마 보정 값 (1.0 이상 권장)

        Returns:
        numpy.ndarray: 노이즈가 추가된 이미지
        """
        rows, cols = src.shape[:2]

        # 고도각을 라디안으로 변환 후 sin 값을 계산
        angle_rad = np.deg2rad(angle)
        sin_alpha = np.sin(angle_rad)

        # 고도각이 낮을 때 안전 장치 추가
        if sin_alpha < 0.1:  # 특정 각도 이하에서는 최소 밝기값으로 설정
            sin_alpha = 0.1

        # 노이즈 강도 계산 (감마 보정을 추가하여 비선형 효과 적용)
        noise_factor = 1 / (sin_alpha ** gamma * intensity + (1 - intensity))

        # 이미지에 노이즈 적용
        noisy_image = src * noise_factor

        # 값의 범위를 0 ~ 255로 클리핑하고 정수형으로 변환
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        return noisy_image