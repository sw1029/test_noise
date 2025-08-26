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
    
    elif method == 'other':
        pass

# py6s를 이용한 보정 + 다른 라이브러리를 이용한 보정을 추가로 구현한다.
# 센서 보정 + random noise correction은 opencv, scikit-image를 통해 구현
# 6s를 이용하여 보정이 불가한 random noise, sensor noise는 opencv, scikit-image를 통해 구현
# sunAngle noise의 경우 absoslute correction으로 진행

# 아래의 denoise_??? 함수들은 py6s를 이용하여 연산을 통해 보정하는 함수들임
# 연산 수식 코드는 utils.py에 위임
def denoise_terrain_6s(src):
    # terrain noise에 속하는 noise를 보정
    # terrain noise 단독
    img = src.copy()
    return img
def denoise_atmospheric_6s(src):
    # atmospheric noise에 속하는 noise를 보정
    # atmospheric noise 단독
    img = src.copy()
    return img
def denoise_sunAngle(src):
    # sun angle noise에 속하는 noise를 보정
    # sun angle noise 단독
    img = src.copy()
    return img
def denoise_sensor(src):
    # sensor noise에 속하는 noise를 보정
    # stripe noise, missing line noise, vignette noise
    img = src.copy()
    return img
def denoise_random(src):
    # random noise에 속하는 noise를 보정
    # salt & pepper noise, gaussian noise, poisson noise 
    img = src.copy()
    return img  



# otb 이용한 보정
def denoise_terrain_other(src):
    # terrain noise에 속하는 noise를 보정
    # terrain noise 단독
    img = src.copy()
    return img
def denoise_atmospheric_other(src):
    # atmospheric noise에 속하는 noise를 보정
    # atmospheric noise 단독
    img = src.copy()
    return img