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
    def add_noise(src,
                  factor=0.1,
                  haze=True, rayleigh=True,
                  yaml_name='KOMPSAT.yaml',
                  sun_angle=30) -> np.ndarray:

        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, '..', '..', 'config', yaml_name) # config 디렉토리에서 yaml을 읽어들인다

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

        ESUN_B = band_params.get('blue', {}).get('ESUN')
        ESUN_G = band_params.get('green', {}).get('ESUN')
        ESUN_R = band_params.get('red', {}).get('ESUN')
        ESUN_NIR = band_params.get('nir', {}).get('ESUN')

        center_B = band_params.get('blue', {}).get('center')
        center_G = band_params.get('green', {}).get('center')
        center_R = band_params.get('red', {}).get('center')
        center_NIR = band_params.get('nir', {}).get('center')

        rows, cols, channels = src.shape # src에서 row, cols, channels 추출
        atmospheric_noise_image = src.copy() # 최종적으로 return 하게 될 객체

        radiance_B = DN2radiance(src[:, :, 0], gain_B, offset_B)
        radiance_G = DN2radiance(src[:, :, 1], gain_G, offset_G)
        radiance_R = DN2radiance(src[:, :, 2], gain_R, offset_R)
        if channels == 4: # NIR 채널이 존재하는 경우 연산
            radiance_NIR = DN2radiance(src[:, :, 3], gain_NIR, offset_NIR)

        solar_zenith= np.deg2rad(90 - sun_angle) # 태양 천정각 계산

        new_radiance_B = radiance_B.copy()
        new_radiance_G = radiance_G.copy()
        new_radiance_R = radiance_R.copy()
        if channels == 4:
            new_radiance_NIR = radiance_NIR.copy()

        '''
        rad0 : Lp
        rad1 : Lp + H * T
        rad1 - rad0 = H * T
        base = (ESUN * cosθ) / (pi * d^2) # cos으로 변환한 이유는 입사각 -> 태양 천정각으로 변환을 수행하였기 때문이다
        radiance / base => 보간된 reflectance(ρ)값
        최종 수식 = (radiance / base) * (rad1 - rad0) + rad0
                = ρ * (H * T) + Lp
        
        이 부분에서 지표 재질에 따라 BRDF가 달라질 수 있다.
        rad0과 rad1을 통해 ρ를 0, 1 로 가정하여 reflectance가 0, 1일때 의 각 radiance를 계산한다.
        reflectance가 0과 1일때인 경우를 계산 후 선형 보간하도록 구현되었으며, 필요 시 지표 재질에 따라 별도 분기를 나누어 구현 가능하다.
        '''

        # haze와 rayleigh가 둘 다 적용된 rad0, rad1
        # 1000.0 으로 나누어 주는 이유는 단위를 맞춰주기 위함이다.
        rad0_both_B, rad1_both_B = get_rad0_rad1(center_B / 1000.0, solar_zenith, haze=True)
        rad0_both_G, rad1_both_G = get_rad0_rad1(center_G / 1000.0, solar_zenith, haze=True)
        rad0_both_R, rad1_both_R = get_rad0_rad1(center_R / 1000.0, solar_zenith, haze=True)
        if channels == 4:
            rad0_both_NIR, rad1_both_NIR = get_rad0_rad1(center_NIR / 1000.0, solar_zenith, haze=True)

        # rayleigh만 적용된 rad0, rad1
        rad0_rayleigh_B, rad1_rayleigh_B = get_rad0_rad1(center_B / 1000.0, solar_zenith, haze=False)
        rad0_rayleigh_G, rad1_rayleigh_G = get_rad0_rad1(center_G / 1000.0, solar_zenith, haze=False)
        rad0_rayleigh_R, rad1_rayleigh_R = get_rad0_rad1(center_R / 1000.0, solar_zenith, haze=False)
        if channels == 4:
            rad0_rayleigh_NIR, rad1_rayleigh_NIR = get_rad0_rad1(center_NIR / 1000.0, solar_zenith, haze=False)

        # haze만 적용된 radiance는 계산 편의상 위에서 구한 값들과 base값들을 통해 산출하는 방식으로 이후 도출한다.

        # base 계산
        baseB = reflectance2radiance(1.0, 1.0, ESUN_B, solar_zenith)
        baseG = reflectance2radiance(1.0, 1.0, ESUN_G, solar_zenith)
        baseR = reflectance2radiance(1.0, 1.0, ESUN_R, solar_zenith)
        if channels == 4:
            baseNIR = reflectance2radiance(1.0, 1.0, ESUN_NIR, solar_zenith)

        # 파라미터로 제공된 haze, rayleigh 활성화 여부에 따라 noise를 다르게 적용하여 반환함

        if haze and rayleigh:
            new_radiance_B = (radiance_B / baseB) * (rad1_both_B - rad0_both_B) + rad0_both_B
            new_radiance_G = (radiance_G / baseG) * (rad1_both_G - rad0_both_G) + rad0_both_G
            new_radiance_R = (radiance_R / baseR) * (rad1_both_R - rad0_both_R) + rad0_both_R
            if channels == 4 :
                new_radiance_NIR = (radiance_NIR / baseNIR) * (rad1_both_NIR - rad0_both_NIR) + rad0_both_NIR
        elif haze and not rayleigh:
            new_radiance_B = ((radiance_B / baseB)
                              * ((rad1_both_B - rad0_both_B) - (rad1_rayleigh_B - rad0_rayleigh_B))
                              + ( rad0_both_B - rad0_rayleigh_B))
            new_radiance_G = ((radiance_G / baseG)
                              * ((rad1_both_G - rad0_both_G) - (rad1_rayleigh_G - rad0_rayleigh_G))
                              + ( rad0_both_G - rad0_rayleigh_G))
            new_radiance_R = ((radiance_R / baseR)
                              * ((rad1_both_R - rad0_both_R) - (rad1_rayleigh_R - rad0_rayleigh_R))
                              + ( rad0_both_R - rad0_rayleigh_R))
            if channels == 4 :
                new_radiance_NIR = ((radiance_NIR / baseNIR)
                                  * ((rad1_both_NIR - rad0_both_NIR) - (rad1_rayleigh_NIR - rad0_rayleigh_NIR))
                                  + ( rad0_both_NIR - rad0_rayleigh_NIR))

        elif not haze and rayleigh:
            new_radiance_B = (radiance_B / baseB) * (rad1_rayleigh_B - rad0_rayleigh_B) + rad0_rayleigh_B
            new_radiance_G = (radiance_G / baseG) * (rad1_rayleigh_G - rad0_rayleigh_G) + rad0_rayleigh_G
            new_radiance_R = (radiance_R / baseR) * (rad1_rayleigh_R - rad0_rayleigh_R) + rad0_rayleigh_R
            if channels == 4 :
                new_radiance_NIR = (radiance_NIR / baseNIR) * (rad1_rayleigh_NIR - rad0_rayleigh_NIR) + rad0_rayleigh_NIR


        # radiance -> DN 변환 후 0~255 범위 내의 값이 되도록 clipping 수행
        atmospheric_noise_image[:, :, 0] = np.clip(radiance2DN(new_radiance_B, gain_B, offset_B),
                                                    0, 255).astype(np.uint8)
        atmospheric_noise_image[:, :, 1] = np.clip(radiance2DN(new_radiance_G, gain_G, offset_G),
                                                   0, 255).astype(np.uint8)
        atmospheric_noise_image[:, :, 2] = np.clip(radiance2DN(new_radiance_R, gain_R, offset_R),
                                                   0, 255).astype(np.uint8)
        if channels == 4:
            atmospheric_noise_image[:, :, 3] = np.clip(radiance2DN(new_radiance_NIR, gain_NIR, offset_NIR),
                                                       0, 255).astype(np.uint8)

        # noise 강도 조절
        atmospheric_noise_image = src * (1 - factor) + atmospheric_noise_image * factor
        atmospheric_noise_image = np.clip(atmospheric_noise_image, 0, 255).astype(np.uint8)

        return atmospheric_noise_image