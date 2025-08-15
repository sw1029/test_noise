'''
업무 1 - image L1, L2를 통해 왜곡 RAW image(L0) 생성
업무 2 - 이미지 간의 noise 정도 비교, 균일화 후 noise별 민감도 측정
업무 3 - 오픈소스 라이브러리를 통해 noise 이미지를 재보정하여 L1, L2와의 원본 유사도 비교

구현해야 하는 기능 - utils

1. noise 정도 비교
2. noise 균일화
3. 이미지의 유사도 비교
4. 노이즈별 민감도 측정(PSNR RMS, MAE, SSIM과 같은 이미지 왜곡 측정 비율 척도)

종류별 noise 구현
    Atmospheric noise - 대기 왜곡
    Radiometric noise - 방사 왜곡
    Sensor Noise/Defect - 카메라 센서 왜곡
    terrian noise - 지형 왜곡
'''

import math
import numpy as np

# radiance <-> reflectance
def radiance2reflectance(radiance, distance, ESUN, solar_angle):
    return (math.pi) * radiance * (distance ** 2) / (ESUN * math.cos(solar_angle))

def reflectance2radiance(reflectance, distance, ESUN, solar_angle):
    return reflectance * ESUN * math.cos(solar_angle) / (math.pi * (distance ** 2))


# radiance <-> DN
def DN2radiance(DN,gain,offset):
    return gain * DN + offset
def radiance2DN(radiance, gain, offset):
    return (radiance - offset) / gain


# Minnaert correction 역연산
def inverse_Minnaert(radiance,i,e,k):
    '''
    i : 태양 입사각
    e : slope
    k : Minnaert 상수

    return : 역연산을 적용한 radiance
    '''
    return radiance / (math.cos(np.deg2rad(e)) ** (k-1) * math.cos(np.deg2rad(i)) ** k)