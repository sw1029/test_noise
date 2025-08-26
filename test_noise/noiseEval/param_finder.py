'''
균일한 노이즈 평가를 위한 파라미터 탐색기
'''

from ..noiseGenerator import *
from .metric import *
import numpy as np
from tqdm import tqdm

NOISE = {
    'gaussian': gaussianNoise,
    'salt_pepper': saltPepperNoise,
    'vignetting': vignettingNoise,
    'missingLine': missingLineNoise,
    'striping': stripingNoise,
    'sunAngle': sunAngleNoise,
    'terrain': terrainNoise,
    'atmospheric': atmosphericNoise,
    'poisson': poissonNoise
}

METRIC = {
    'psnr': {'func': psnr, 'positive': False},
    'rmse': {'func': rmse, 'positive': True},
    'mae': {'func': mae, 'positive': True},
    'ssim': {'func': ssim, 'positive': False}
}

PARAMS = {
    'gaussian': ('var', 0, 10000),
    'salt_pepper': ('amount', 0.0, 1.0),
    'vignetting': ('strength', 0.0, 1.0),
    'missingLine': ('num_threshold', 1, 100), # 다른 파라미터는 기본값 사용
    'striping': ('noise_strength', 1, 50),
    'sunAngle': ('intensity', 0.0, 1.0),
    'terrain': ('factor', 0.0, 1.0),
    'atmospheric': ('factor', 0.0, 1.0),
    'poisson': ('factor', 0.0, 1.0)
}

def find_param(img, target, 
               noise_type = 'gaussian',value_type = "rmse", 
               iter=100 , tol=0.01):
    '''
    이진 탐색을 통하여 선택한 noise의 target_value에 도달하는 파라미터를 찾는다.
    '''
    noise = NOISE[noise_type]
    metric = METRIC[value_type]
    param_name, min, max = PARAMS[noise_type]

    positive = metric['positive']

    for _ in tqdm(range(iter), desc = f"<<finding {noise_type}>> target value:{target}"):
        mid = (min + max) / 2
        noisy_img = noise(img, **{param_name: mid})
        value = metric['func'](img, noisy_img)

        if abs(value - target) < tol:
            return mid , value

        if (positive and value < target) or (not positive and value > target):
            min = mid
        else:
            max = mid
    return (min + max) / 2 , value