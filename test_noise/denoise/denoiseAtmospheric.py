from .base import Denoise
from ..utils import *
import numpy as np
import os
import yaml

class DenoiseAtmospheric(Denoise):
    @staticmethod
    def denoise(src, factor = 0.3,
                  haze=True, rayleigh=True,
                  yaml_name='KOMPSAT.yaml',
                  sun_angle=30,
                  calib_mode: str = 'auto') -> np.ndarray:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, '..', 'config', yaml_name) # config 디렉토리에서 yaml을 읽어들인다

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

        use_reflectance = (calib_mode == 'reflectance') or (calib_mode == 'auto' and src.dtype == np.uint8)

        if not use_reflectance:
            radiance_B = DN2radiance(src[:, :, 0], gain_B, offset_B)
            radiance_G = DN2radiance(src[:, :, 1], gain_G, offset_G)
            radiance_R = DN2radiance(src[:, :, 2], gain_R, offset_R)
            if channels == 4: # NIR 채널이 존재하는 경우 연산
                radiance_NIR = DN2radiance(src[:, :, 3], gain_NIR, offset_NIR)

        solar_zenith= np.deg2rad(90 - sun_angle) # 태양 천정각 계산

        '''
        rad0 : Lp
        rad1 : Lp + H * T
        rad1 - rad0 = H * T
        base = (ESUN * cosθ) / (pi * d^2) # cos으로 변환한 이유는 입사각 -> 태양 천정각으로 변환을 수행하였기 때문이다
        radiance / base => 보간된 reflectance(ρ)값
        noise 생성 최종 수식 = (radiance / base) * (rad1 - rad0) + rad0
                = ρ * (H * T) + Lp
        noise 보정 최종 수식 = ()
        
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
        
        eps = 1e-6
        if use_reflectance:
            # 반사도 도메인에서 역연산 수행
            rho_mix_B = src[:, :, 0].astype(np.float32) / 255.0
            rho_mix_G = src[:, :, 1].astype(np.float32) / 255.0
            rho_mix_R = src[:, :, 2].astype(np.float32) / 255.0
            if channels == 4:
                rho_mix_NIR = src[:, :, 3].astype(np.float32) / 255.0

            # Py6S radiance를 반사도 단위로 변환
            d_both_B = (rad1_both_B - rad0_both_B) / baseB; c_both_B = (rad0_both_B / baseB)
            d_both_G = (rad1_both_G - rad0_both_G) / baseG; c_both_G = (rad0_both_G / baseG)
            d_both_R = (rad1_both_R - rad0_both_R) / baseR; c_both_R = (rad0_both_R / baseR)
            if channels == 4:
                d_both_NIR = (rad1_both_NIR - rad0_both_NIR) / baseNIR; c_both_NIR = (rad0_both_NIR / baseNIR)

            d_ray_B = (rad1_rayleigh_B - rad0_rayleigh_B) / baseB; c_ray_B = (rad0_rayleigh_B / baseB)
            d_ray_G = (rad1_rayleigh_G - rad0_rayleigh_G) / baseG; c_ray_G = (rad0_rayleigh_G / baseG)
            d_ray_R = (rad1_rayleigh_R - rad0_rayleigh_R) / baseR; c_ray_R = (rad0_rayleigh_R / baseR)
            if channels == 4:
                d_ray_NIR = (rad1_rayleigh_NIR - rad0_rayleigh_NIR) / baseNIR; c_ray_NIR = (rad0_rayleigh_NIR / baseNIR)

            if haze and rayleigh:
                den_B = np.maximum((1 - factor) + factor * d_both_B, eps)
                den_G = np.maximum((1 - factor) + factor * d_both_G, eps)
                den_R = np.maximum((1 - factor) + factor * d_both_R, eps)
                rho_src_B = (rho_mix_B - factor * c_both_B) / den_B
                rho_src_G = (rho_mix_G - factor * c_both_G) / den_G
                rho_src_R = (rho_mix_R - factor * c_both_R) / den_R
                if channels == 4:
                    den_NIR = np.maximum((1 - factor) + factor * d_both_NIR, eps)
                    rho_src_NIR = (rho_mix_NIR - factor * c_both_NIR) / den_NIR
            elif haze and not rayleigh:
                den_B = np.maximum((1 - factor) + factor * (d_both_B - d_ray_B), eps)
                den_G = np.maximum((1 - factor) + factor * (d_both_G - d_ray_G), eps)
                den_R = np.maximum((1 - factor) + factor * (d_both_R - d_ray_R), eps)
                rho_src_B = (rho_mix_B - factor * (c_both_B - c_ray_B)) / den_B
                rho_src_G = (rho_mix_G - factor * (c_both_G - c_ray_G)) / den_G
                rho_src_R = (rho_mix_R - factor * (c_both_R - c_ray_R)) / den_R
                if channels == 4:
                    den_NIR = np.maximum((1 - factor) + factor * (d_both_NIR - d_ray_NIR), eps)
                    rho_src_NIR = (rho_mix_NIR - factor * (c_both_NIR - c_ray_NIR)) / den_NIR
            elif (not haze) and rayleigh:
                den_B = np.maximum((1 - factor) + factor * d_ray_B, eps)
                den_G = np.maximum((1 - factor) + factor * d_ray_G, eps)
                den_R = np.maximum((1 - factor) + factor * d_ray_R, eps)
                rho_src_B = (rho_mix_B - factor * c_ray_B) / den_B
                rho_src_G = (rho_mix_G - factor * c_ray_G) / den_G
                rho_src_R = (rho_mix_R - factor * c_ray_R) / den_R
                if channels == 4:
                    den_NIR = np.maximum((1 - factor) + factor * d_ray_NIR, eps)
                    rho_src_NIR = (rho_mix_NIR - factor * c_ray_NIR) / den_NIR
            else:
                # haze=False, rayleigh=False 인 경우: 변화 없음 (혼합/역혼합 항 모두 항등)
                rho_src_B = rho_mix_B
                rho_src_G = rho_mix_G
                rho_src_R = rho_mix_R
                if channels == 4:
                    rho_src_NIR = rho_mix_NIR

            # 물리 범위 보장: 역복원된 반사도 0~1로 클램프 후 DN 변환
            rho_src_B = np.clip(rho_src_B, 0.0, 1.0)
            rho_src_G = np.clip(rho_src_G, 0.0, 1.0)
            rho_src_R = np.clip(rho_src_R, 0.0, 1.0)
            atmospheric_noise_image[:, :, 0] = np.clip(rho_src_B * 255.0, 0, 255).astype(np.uint8)
            atmospheric_noise_image[:, :, 1] = np.clip(rho_src_G * 255.0, 0, 255).astype(np.uint8)
            atmospheric_noise_image[:, :, 2] = np.clip(rho_src_R * 255.0, 0, 255).astype(np.uint8)
            if channels == 4:
                rho_src_NIR = np.clip(rho_src_NIR, 0.0, 1.0)
                atmospheric_noise_image[:, :, 3] = np.clip(rho_src_NIR * 255.0, 0, 255).astype(np.uint8)
        else:
            if haze and rayleigh:
                denominator_B = (1 - factor) + (factor / baseB) * (rad1_both_B - rad0_both_B)
                denominator_B = np.maximum(denominator_B, eps)
                new_radiance_B = (radiance_B - factor * rad0_both_B) / denominator_B

                denominator_G = (1 - factor) + (factor / baseG) * (rad1_both_G - rad0_both_G)
                denominator_G = np.maximum(denominator_G, eps)
                new_radiance_G = (radiance_G - factor * rad0_both_G) / denominator_G

                denominator_R = (1 - factor) + (factor / baseR) * (rad1_both_R - rad0_both_R)
                denominator_R = np.maximum(denominator_R, eps)
                new_radiance_R = (radiance_R - factor * rad0_both_R) / denominator_R
                
                if channels == 4:
                    denominator_NIR = (1 - factor) + (factor / baseNIR) * (rad1_both_NIR - rad0_both_NIR)
                    denominator_NIR = np.maximum(denominator_NIR, eps)
                    new_radiance_NIR = (radiance_NIR - factor * rad0_both_NIR) / denominator_NIR

            elif haze and not rayleigh:
                denominator_B = (1 - factor) + (factor / baseB) * ((rad1_both_B - rad0_both_B) - (rad1_rayleigh_B - rad0_rayleigh_B))
                denominator_B = np.maximum(denominator_B, eps)
                new_radiance_B = (radiance_B - factor * (rad0_both_B - rad0_rayleigh_B)) / denominator_B

                denominator_G = (1 - factor) + (factor / baseG) * ((rad1_both_G - rad0_both_G) - (rad1_rayleigh_G - rad0_rayleigh_G))
                denominator_G = np.maximum(denominator_G, eps)
                new_radiance_G = (radiance_G - factor * (rad0_both_G - rad0_rayleigh_G)) / denominator_G

                denominator_R = (1 - factor) + (factor / baseR) * ((rad1_both_R - rad0_both_R) - (rad1_rayleigh_R - rad0_rayleigh_R))
                denominator_R = np.maximum(denominator_R, eps)
                new_radiance_R = (radiance_R - factor * (rad0_both_R - rad0_rayleigh_R)) / denominator_R
                
                if channels == 4:
                    denominator_NIR = (1 - factor) + (factor / baseNIR) * ((rad1_both_NIR - rad0_both_NIR) - (rad1_rayleigh_NIR - rad0_rayleigh_NIR))
                    denominator_NIR = np.maximum(denominator_NIR, eps)
                    new_radiance_NIR = (radiance_NIR - factor * (rad0_both_NIR - rad0_rayleigh_NIR)) / denominator_NIR

            elif not haze and rayleigh:
                denominator_B = (1 - factor) + (factor / baseB) * (rad1_rayleigh_B - rad0_rayleigh_B)
                denominator_B = np.maximum(denominator_B, eps)
                new_radiance_B = (radiance_B - factor * rad0_rayleigh_B) / denominator_B

                denominator_G = (1 - factor) + (factor / baseG) * (rad1_rayleigh_G - rad0_rayleigh_G)
                denominator_G = np.maximum(denominator_G, eps)
                new_radiance_G = (radiance_G - factor * rad0_rayleigh_G) / denominator_G

                denominator_R = (1 - factor) + (factor / baseR) * (rad1_rayleigh_R - rad0_rayleigh_R)
                denominator_R = np.maximum(denominator_R, eps)
                new_radiance_R = (radiance_R - factor * rad0_rayleigh_R) / denominator_R
                
                if channels == 4:
                    denominator_NIR = (1 - factor) + (factor / baseNIR) * (rad1_rayleigh_NIR - rad0_rayleigh_NIR)
                    denominator_NIR = np.maximum(denominator_NIR, eps)
                    new_radiance_NIR = (radiance_NIR - factor * rad0_rayleigh_NIR) / denominator_NIR


            # radiance -> DN 변환 후 0~255 범위 내의 값이 되도록 clipping 수행, uint8로 명시 변환
            atmospheric_noise_image[:, :, 0] = np.clip(
                radiance2DN(new_radiance_B, gain_B, offset_B), 0, 255
            ).astype(np.uint8)
            atmospheric_noise_image[:, :, 1] = np.clip(
                radiance2DN(new_radiance_G, gain_G, offset_G), 0, 255
            ).astype(np.uint8)
            atmospheric_noise_image[:, :, 2] = np.clip(
                radiance2DN(new_radiance_R, gain_R, offset_R), 0, 255
            ).astype(np.uint8)
            if channels == 4:
                atmospheric_noise_image[:, :, 3] = np.clip(
                    radiance2DN(new_radiance_NIR, gain_NIR, offset_NIR), 0, 255
                ).astype(np.uint8)

        return atmospheric_noise_image
