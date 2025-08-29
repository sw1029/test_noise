import rasterio
import numpy as np


# RGB GeoTIFF
with rasterio.open("data/sample_rgb.tif") as src:
    rgb_array = src.read()  # (bands, H, W) 형태
print(rgb_array.shape)  # 예: (3, 512, 512)

# DEM GeoTIFF
with rasterio.open("data/sample_dem.tif") as src:
    dem_array = src.read(1)  # 첫 번째 밴드만 (H, W)
print(dem_array.shape)  # 예: (512, 512)

# RGB GeoTIFF → OpenCV 스타일 (H, W, C)
rgb_cv2 = np.transpose(rgb_array, (1, 2, 0))  # (H, W, C)

# DEM은 단일 채널이므로 그대로 사용 가능 (H, W)
dem_cv2 = dem_array

from test_noise.noiseGenerator import *
from test_noise.noiseEval import *
from test_noise.denoise import *
import cv2
import os
import pandas as pd

import cv2

# rgb_cv2: (H, W, 3), RGB 순서
bgr_cv2 = cv2.cvtColor(rgb_cv2, cv2.COLOR_RGB2BGR)
terrain_noised_image = terrainNoise(bgr_cv2, DEM= dem_cv2, factor=0.3)
terrain_denoised_image = terrain(terrain_noised_image, DEM=dem_cv2)

cv2.imshow('Terrain Noised Image', terrain_noised_image)
cv2.imshow('Terrain denoised Image', terrain_denoised_image)
cv2.imshow('bgr', bgr_cv2)

cv2.waitKey(0)