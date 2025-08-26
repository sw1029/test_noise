from .base import NoiseBase
import numpy as np
import cv2

class VignettingNoise(NoiseBase):
    @staticmethod
    def add_noise(src, strength=0.4, power=2.3):
        rows, cols, channels = src.shape
        # strength : 비네팅 강도 (0 ~ 1 사이의 값, 높을수록 강함)
        # power : 거리의 지수 적용 값 (높을수록 급격히 어두워짐)
        # 거리 기반 마스크 생성 (중앙에서 가장자리로 갈수록 값이 작아짐)
        X_result, Y_result = np.meshgrid(np.linspace(-1, 1, cols), np.linspace(-1, 1,rows))
        distance = np.sqrt(X_result ** 2 + Y_result ** 2)

        # 거리의 지수화를 통해 가장자리와 중앙의 차이를 크게 만듦
        mask = 1 - (distance ** power * strength)
        mask = np.clip(mask, 0, 1)

        # 컬러 이미지인 경우 마스크를 3채널로 확장
        if len(src.shape) == 3 and src.shape[2] == 3:
            mask = cv2.merge([mask] * 3)

        # 비네팅 마스크를 이미지에 곱해서 가장자리 어둡게
        vignetting_image = src * mask
        return np.clip(vignetting_image, 0, 255).astype(np.uint8)