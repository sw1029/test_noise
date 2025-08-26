from .base import Denoise
import numpy as np

class DenoiseSunAngle(Denoise):
    @staticmethod
    def denoise(src: np.ndarray, sunAngle = 45, intensity=0.5, gamma=1.0) -> np.ndarray:
        '''
        입력된 이미지에 노이즈를 보정하는 부분입니다.
        init에서 입력받은 이미지에 noise를 보정한 후 반환합니다. 
        sun angle noise를 보정합니다.
        '''
        img = src.copy()      
        alpha = np.sin(np.deg2rad(sunAngle)) # 태양 고도각의 사인값 계산

        # 고도각이 낮을 때 안전 장치 추가
        if alpha < 0.1:  # 특정 각도 이하에서는 최소 밝기값으로 설정
            alpha = 0.1

        # 보정 계수 계산 및 적용
        factor = (alpha ** gamma * intensity + (1 - intensity))
        img = img * factor

        img = np.clip(img, 0, 255).astype(np.uint8)
        return img