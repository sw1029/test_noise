from .base import Denoise
import numpy as np
from skimage.restoration import inpaint_biharmonic

class DenoiseMissingLine(Denoise):
    @staticmethod
    def denoise(src: np.ndarray) -> np.ndarray:
        '''
        입력된 이미지에 노이즈를 보정하는 부분입니다.
        init에서 입력받은 이미지에 noise를 보정한 후 반환합니다. 
        missingLine noise에 속하는 noise를 보정합니다.

        참고 레퍼런스
        - https://www.tutorialspoint.com/scikit-image/scikit-image-image-inpainting.html
        '''
        img = src.copy()
        
        mask = mask = (img == 0).all(axis=2)
        img = inpaint_biharmonic(img, mask, channel_axis=-1)
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        
        return img