from .base import Denoise
import numpy as np
import cv2


class DenoiseRandom(Denoise):
    @staticmethod
    def denoise(src: np.ndarray, type='gaussian') -> np.ndarray:
        '''
        입력된 이미지에 노이즈를 보정하는 부분입니다.
        init에서 입력받은 이미지에 noise를 보정한 후 반환합니다. 
        random noise에 속하는 noise를 보정합니다.
        - random noise : salt & pepper noise, gaussian noise, poisson noise

        참고 레퍼런스
        - https://medium.com/@abhishekjainindore24/salt-and-pepper-noise-and-how-to-remove-it-in-machine-learning-032d76b138a5
        - https://www.kaggle.com/code/chanduanilkumar/adding-and-removing-image-noise-in-python
        - https://en.wikipedia.org/wiki/Anscombe_transform
        '''
        img = src.copy()

        if type == 'gaussian':
            img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

        elif type == 'salt & pepper':
            img = cv2.medianBlur(img, 3)
        
        elif type == 'poisson':
            img = 2 * np.sqrt(img.astype(np.float32) + 3/8)# Anscombe transform을 통해 gaussian noise에 대한 denoise로 해석
            max = img.max() 
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

            img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

            img = cv2.normalize(img.astype(np.float32), None, 0, max, cv2.NORM_MINMAX)
            img = (img / 2)**2 - 3/8
            img = np.clip(img, 0, 255).astype(np.uint8)
        return img