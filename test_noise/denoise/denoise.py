'''
오픈소스로 제공된 noise 전처리 라이브러리 혹은 프로그램을 통해 다른 모듈에서 생성된 noise 이미지를 재보정
'''

import Py6S
import numpy as np
import cv2
import os
from test_noise.utils import *

def denoise(src, method='py6s'):
    '''
    아래에 구현된 denoise 전용 기능들을 호출해 denoise를 수행하는 함수
    ''' 
    img = src.copy()
    
    if method == 'py6s':
        pass
    
    elif method == 'otb':
        pass