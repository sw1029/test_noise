from .base import Denoise
from ..utils import *
import numpy as np
import yaml
import os

class DenoiseTerrain(Denoise):
    @staticmethod
    def denoise(src, factor = 0.3,
                  sun_angle=30, 
                  DEM=None, pixel_size=1.0,
                  slope=30, 
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
        terrain_denoise_image = src.copy()

        if DEM is not None:
            px, py = np.gradient(DEM, pixel_size)
            _slope = np.arctan(np.sqrt(px**2 + py**2))
            _slope = np.rad2deg(_slope) 
        else : _slope = slope     

        radiance_B = np.clip(DN2radiance(src[:,:,0], gain_B, offset_B), 0, None)
        radiance_G = np.clip(DN2radiance(src[:,:,1], gain_G, offset_G), 0, None)
        radiance_R = np.clip(DN2radiance(src[:,:,2], gain_R, offset_R), 0, None)
        
        radiance_B = DN2radiance(src[:,:,0], gain_B, offset_B)
        radiance_G = DN2radiance(src[:,:,1], gain_G, offset_G)
        radiance_R = DN2radiance(src[:,:,2], gain_R, offset_R)

        terrain_denoise_image[:, :, 0] = Minnaert(radiance_B, sun_angle, _slope, Minnaert_constant_B)
        terrain_denoise_image[:, :, 1] = Minnaert(radiance_G, sun_angle, _slope, Minnaert_constant_G)
        terrain_denoise_image[:, :, 2] = Minnaert(radiance_R, sun_angle, _slope, Minnaert_constant_R)

        terrain_denoise_image[:, :, 0] = radiance2DN(terrain_denoise_image[:, :, 0], gain_B, offset_B)
        terrain_denoise_image[:, :, 1] = radiance2DN(terrain_denoise_image[:, :, 1], gain_G, offset_G)
        terrain_denoise_image[:, :, 2] = radiance2DN(terrain_denoise_image[:, :, 2], gain_R, offset_R)
        
        if channels == 4:
            radiance_NIR = DN2radiance(src[:, :, 3], gain_NIR, offset_NIR)
            terrain_denoise_image[:, :, 3] = Minnaert(radiance_NIR, sun_angle, _slope, Minnaert_constant_NIR)
            terrain_denoise_image[:, :, 3] = radiance2DN(terrain_denoise_image[:, :, 3], gain_NIR, offset_NIR)
       
        terrain_denoise_image = np.clip(terrain_denoise_image, 0, 255).astype(np.uint8)
        return terrain_denoise_image