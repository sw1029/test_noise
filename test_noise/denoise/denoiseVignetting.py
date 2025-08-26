from .base import Denoise
import numpy as np
import cv2

class DenoiseVignetting(Denoise):
    @staticmethod
    def denoise(src: np.ndarray, strength=0.4, power=2.3) -> np.ndarray:
        '''
        입력된 이미지에 노이즈를 보정하는 부분입니다.
        init에서 입력받은 이미지에 noise를 보정한 후 반환합니다. 
        vignetting noise에 속하는 noise를 보정합니다.
        기존의 vignetting noise의 mask 연산은 동일하게 수행하되, 원본 이미지에 나누는 방식으로 연산됩니다.

        참고 레퍼런스
        - https://answers.opencv.org/question/231896/vignetting-correction/
        '''
        img = src.copy()
        rows, cols, _ = img.shape

        X_result, Y_result = np.meshgrid(np.linspace(-1, 1, cols), np.linspace(-1, 1,rows))
        distance = np.sqrt(X_result ** 2 + Y_result ** 2)

        # 거리의 지수화를 통해 가장자리와 중앙의 차이를 크게 만듦
        mask = 1 - (distance ** power * strength)
        mask = np.clip(mask, 0, 1)

        # 컬러 이미지인 경우 마스크를 3채널로 확장
        if len(src.shape) == 3 and src.shape[2] == 3:
            mask = cv2.merge([mask] * 3)
        
        # 비네팅 마스크를 이미지에 나누어서 보정
        img = img / mask
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img