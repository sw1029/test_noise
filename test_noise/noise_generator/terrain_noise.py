from .base import Noise
import numpy as np
from ..utils import inverse_Minnaert, DN2radiance, radiance2DN
import yaml
import os

'''
terrain noise는 Minnaert correction의 역연산을 취하여 구현.
필요 값: radiance, slope, sun angle, Minnaert 상수
이 중 Minnaert 상수와 slope는 임의의 값으로 설정하며, 파라미터 주입을 통하여 조정 가능하도록 구현하였음.
'''

class TerrainNoise(Noise):
    @staticmethod
    def add_noise(src,
                  sun_angle=30, factor=0.1, slope=30, 
                  Minnaert_constant_NIR=0.6,
                  Minnaert_constant_R=0.5,
                  Minnaert_constant_G=0.4,
                  Minnaert_constant_B=0.3,
                  yaml_name="KOMPSAT.yaml") -> np.ndarray:
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, '..', 'config', yaml_name)
        
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
        terrain_noise_image = src.copy()

        radiance_B = DN2radiance(src[:,:,0], gain_B, offset_B)
        radiance_G = DN2radiance(src[:,:,1], gain_G, offset_G)
        radiance_R = DN2radiance(src[:,:,2], gain_R, offset_R)
        
        terrain_noise_image[:, :, 0] = inverse_Minnaert(radiance_B, sun_angle, slope, Minnaert_constant_B)
        terrain_noise_image[:, :, 0] = radiance2DN(terrain_noise_image[:, :, 0], gain_B, offset_B)

        terrain_noise_image[:, :, 1] = inverse_Minnaert(radiance_G, sun_angle, slope, Minnaert_constant_G)
        terrain_noise_image[:, :, 1] = radiance2DN(terrain_noise_image[:, :, 1], gain_G, offset_G)

        terrain_noise_image[:, :, 2] = inverse_Minnaert(radiance_R, sun_angle, slope, Minnaert_constant_R)
        terrain_noise_image[:, :, 2] = radiance2DN(terrain_noise_image[:, :, 2], gain_R, offset_R)
        
        if channels == 4:
            radiance_NIR = DN2radiance(src[:, :, 3], gain_NIR, offset_NIR)
            terrain_noise_image[:, :, 3] = inverse_Minnaert(radiance_NIR, sun_angle, slope, Minnaert_constant_NIR)
            terrain_noise_image[:, :, 3] = radiance2DN(terrain_noise_image[:, :, 3], gain_NIR, offset_NIR)

        # 노이즈 강도 조절
        terrain_noise_image = src * (1 - factor) + terrain_noise_image * factor
        terrain_noise_image = np.clip(terrain_noise_image * factor, 0, 255).astype(np.uint8)
        return terrain_noise_image