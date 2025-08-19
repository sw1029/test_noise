from base import Noise
import numpy as np
import os
from utils import *
import yaml
import Py6S

'''
haze noise, raylaigh noise를 구현하는 class
py6s를 사용하는 방식으로 구현
Ls = H * ρ * T + Lp 에서의 H(total downwelling된 radiance)를 reflectance를 구하는 수식인 
(pi * radiance * d^2) / (ESUN * sinθ)의 역연산을 통해 구하고, T값과 Lp값은 py6s를 통해 계산하는 방식으로 역연산
'''

class AtmosphericNoise(Noise):
    @staticmethod
    def add_noise(src,haze=True, rayleigh=True,yaml_name = 'KOMPSAT.yaml') -> np.ndarray:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, '..', '..', 'config', yaml_name)

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        band_params = config.get('band', {})

        gain_B = band_params.get('blue', {}).get('gain')
        offset_B = band_params.get('blue', {}).get('offset')
        gain_G = band_params.get('green', {}).get('gain')
        offset_G = band_params.get('green', {}).get('offset')
        gain_R = band_params.get('red', {}).get('gain')
        offset_R = band_params.get('red', {}).get('offset')
        gain_NIR = band_params.get('nir', {}).get('gain')
        offset_NIR = band_params.get('nir', {}).get('offset')

        rows, cols, channels = src.shape
        atmospheric_noise_image = src.copy()

        radiance_B = DN2radiance(src[:, :, 0], gain_B, offset_B)
        radiance_G = DN2radiance(src[:, :, 1], gain_G, offset_G)
        radiance_R = DN2radiance(src[:, :, 2], gain_R, offset_R)
        if channels == 4: radiance_NIR = DN2radiance(src[:, :, 3], gain_NIR, offset_NIR)

        haze_noise = None
        rayleigh_noise = None #반환하고자 하는 순수 noise

        if haze: # haze 적용을 True로 지정한 경우
            pass

        if rayleigh: # rayleigh 적용을 True로 지정한 경우
            pass

        return atmospheric_noise_image