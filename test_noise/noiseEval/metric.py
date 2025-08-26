from ..noiseGenerator import *
import numpy as np
from skimage.metrics import structural_similarity as ss

def psnr(origin, noisy):
    origin = origin.astype(np.float64)
    noisy = noisy.astype(np.float64)
    
    mse = np.mean((origin - noisy) ** 2)

    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr_value

def rmse(origin, noisy):
    origin = origin.astype(np.float64)
    noisy = noisy.astype(np.float64)
    
    mse = np.mean((origin - noisy) ** 2)
    rmse_value = np.sqrt(mse)
    return rmse_value

def mae(origin, noisy):
    origin = origin.astype(np.float64)
    noisy = noisy.astype(np.float64)
    
    mae_value = np.mean(np.abs(origin - noisy))
    return mae_value

def ssim(origin, noisy):
    ssim_value = ss(origin, noisy, channel_axis=2, data_range=origin.max() - origin.min())
    return ssim_value