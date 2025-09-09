from test_noise.noiseEval.evaluator import make_param_csv
from test_noise.noiseGenerator import *
from test_noise.noiseEval import *
from test_noise.denoise import *
from test_noise.utils import dis

import cv2
import os
import pandas as pd
import rasterio
import numpy as np

image_path = 'input_images/P0000__512__2304___1536.png'  # 원본 이미지 경로
noisy_dir = 'output/noisy'
denoised_dir = 'output/denoised'
csv_dir = 'output/csv'

# 출력 디렉토리 생성
os.makedirs(noisy_dir, exist_ok=True)
os.makedirs(denoised_dir, exist_ok=True)
os.makedirs(csv_dir, exist_ok=True)

with rasterio.open("data/sample_rgb.tif") as src:
    rgb_array = src.read()  # (bands, H, W) 형태
print(rgb_array.shape)  
# DEM GeoTIFF
with rasterio.open("data/sample_dem.tif") as src:
    dem_array = src.read(1)  # 첫 번째 밴드만 (H, W)
print(dem_array.shape)

# RGB GeoTIFF → OpenCV 스타일 (H, W, C)
rgb_cv2 = np.transpose(rgb_array, (1, 2, 0))  # (H, W, C)
# DEM은 단일 채널이므로 그대로 사용 가능 (H, W)
dem_cv2 = dem_array

# nan 값이 있는지 확인하고, 있다면 0으로 대체
if np.isnan(rgb_cv2).any():
    rgb_cv2 = np.nan_to_num(rgb_cv2, nan=0.0)
if np.isnan(dem_array).any():
    dem_array = np.nan_to_num(dem_array, nan=0.0)
img = cv2.cvtColor(rgb_cv2.astype(np.float32), cv2.COLOR_RGB2BGR)

# TIF 원본은 부동소수 스케일일 수 있으므로 표시/연산용으로 uint8 정규화 버전 생성
tif_src = dis(img)
tif_dem = dem_cv2  # 아래에서 dem_cv2가 None으로 변하기 전 보관
cv2.imwrite(os.path.join(noisy_dir, 'tif_original.png'), tif_src)

src = cv2.imread(image_path) #필요시 테스트용으로 아래의 주석과 함께 활성화.
dem_cv2 = None

# Noise 추가
terrain_noised_image = terrainNoise(src, factor=0.7, DEM=dem_cv2)
atmosphric_noised_image = atmosphericNoise(src, factor=0.7)
gaussian_noised_image = gaussianNoise(src)
missing_line_noised_image = missingLineNoise(src)
salt_pepper_noised_image = saltPepperNoise(src)
poisson_noised_image = poissonNoise(src)
striping_noised_image = stripingNoise(src, direction = 'vertical')
sun_angle_noised_image = sunAngleNoise(src, intensity= 1.0)
vignetting_noised_image = vignettingNoise(src)

# Denoise
# 지형 보정: gain/offset이 샘플 이미지에는 없으므로 C-correction 기반 Topo 보정 사용
# 색 틀어짐 방지를 위해 radiance 기반 단일 스케일, 보정 강도 제한
if dem_cv2 is None:
    # DEM이 없으면 Minnaert 기반 보정 경로 사용
    terrain_denoised_image = terrain(
        terrain_noised_image
    )
else:
    # DEM이 있으면 C-correction/Minnaert topo 보정 경로 사용
    terrain_denoised_image = terrainTopo(
        terrain_noised_image,
        DEM=dem_cv2,
        sun_azimuth=225,
        sun_elevation=45,
        method="minnaert",
        mode="luminance",
        scale_clip=(0.8, 1.2),
        shadow_thresh=0.2,
        robust=True,
        ransac_iter=300,
        inlier_sigma=2.0
    )
atmospheric_denoised_image = atmospher(atmosphric_noised_image, factor=0.3, haze=True, rayleigh=True, yaml_name='KOMPSAT.yaml', sun_angle=30)
gaussian_denoised_image = random(gaussian_noised_image, type='gaussian')
missing_denoised_image = missingLine(missing_line_noised_image)
salt_pepper_denoised_image = random(salt_pepper_noised_image, type='saltPepper')
poisson_denoised_image = random(poisson_noised_image, type='poisson')
striping_denoised_image = stripe(striping_noised_image)
sun_angle_denoised_image = sunAngle(sun_angle_noised_image)
vignetting_denoised_image = vignetting(vignetting_noised_image)


denoised = {
    "Terrain": terrain_denoised_image,
    "Atmospheric": atmospheric_denoised_image,
    "Gaussian": gaussian_denoised_image,
    "Missing Line": missing_denoised_image,
    "Salt & Pepper": salt_pepper_denoised_image,
    "Poisson": poisson_denoised_image,
    "Striping": striping_denoised_image,
    "Sun Angle": sun_angle_denoised_image,
    "Vignetting": vignetting_denoised_image
}

noisy = {
    "Terrain": terrain_noised_image,
    "Atmospheric": atmosphric_noised_image,
    "Gaussian": gaussian_noised_image,
    "Missing Line": missing_line_noised_image,
    "Salt & Pepper": salt_pepper_noised_image,
    "Poisson": poisson_noised_image,
    "Striping": striping_noised_image,
    "Sun Angle": sun_angle_noised_image,
    "Vignetting": vignetting_noised_image
}

origin = src

def compute_metrics(a, b):
    a_disp = dis(a)
    b_disp = dis(b)
    return {
        "SSIM": ssim(a_disp, b_disp),
        "PSNR": psnr(a_disp, b_disp),
        "RMSE": rmse(a_disp, b_disp),
        "MAE": mae(a_disp, b_disp),
    }

origin_disp = dis(origin)

print("=== Similarity to Original (Noisy images) ===")
for name, noisy_img in noisy.items():
    m = compute_metrics(origin_disp, noisy_img)
    print(f"Noisy {name}: SSIM={m['SSIM']:.4f}, PSNR={m['PSNR']:.2f}, RMSE={m['RMSE']:.2f}, MAE={m['MAE']:.2f}")

print("=== Similarity to Original (Denoised images) ===")
for name, denoised_img in denoised.items():
    m = compute_metrics(origin_disp, denoised_img)
    print(f"Denoised {name}: SSIM={m['SSIM']:.4f}, PSNR={m['PSNR']:.2f}, RMSE={m['RMSE']:.2f}, MAE={m['MAE']:.2f}")

# Build metrics summary DataFrames and save
rows = []
improvements = []
for name in noisy.keys():
    nimg = noisy[name]
    dimg = denoised[name]

    m_on = compute_metrics(origin_disp, nimg)   # Original vs Noisy
    m_od = compute_metrics(origin_disp, dimg)   # Original vs Denoised
    m_nd = compute_metrics(nimg, dimg)          # Noisy vs Denoised

    rows.append({
        "Noise": name, "Pair": "Original vs Noisy",
        "SSIM": m_on["SSIM"], "PSNR": m_on["PSNR"], "RMSE": m_on["RMSE"], "MAE": m_on["MAE"],
    })
    rows.append({
        "Noise": name, "Pair": "Original vs Denoised",
        "SSIM": m_od["SSIM"], "PSNR": m_od["PSNR"], "RMSE": m_od["RMSE"], "MAE": m_od["MAE"],
    })
    rows.append({
        "Noise": name, "Pair": "Noisy vs Denoised",
        "SSIM": m_nd["SSIM"], "PSNR": m_nd["PSNR"], "RMSE": m_nd["RMSE"], "MAE": m_nd["MAE"],
    })

    improvements.append({
        "Noise": name,
        "dSSIM": m_od["SSIM"] - m_on["SSIM"],
        "dPSNR": m_od["PSNR"] - m_on["PSNR"],
        "dRMSE": m_on["RMSE"] - m_od["RMSE"],  # positive is better
        "dMAE": m_on["MAE"] - m_od["MAE"],      # positive is better
    })

metrics_df = pd.DataFrame(rows)
improve_df = pd.DataFrame(improvements)

metrics_csv = os.path.join(csv_dir, 'metrics_summary.csv')
improve_csv = os.path.join(csv_dir, 'metrics_improvement.csv')
metrics_df.to_csv(metrics_csv, index=False)
improve_df.to_csv(improve_csv, index=False)

# ==========================
# TIF 입력에 대한 Noise/Denoise 및 지표 저장
# ==========================
print("=== Processing TIF inputs (with DEM if available) ===")

# Noise 추가 (TIF)
tif_terrain_noised_image = terrainNoise(tif_src, factor=0.7, DEM=tif_dem)
tif_atmosphric_noised_image = atmosphericNoise(tif_src, factor=0.7)
tif_gaussian_noised_image = gaussianNoise(tif_src, mean = 7, var = 70)
tif_missing_line_noised_image = missingLineNoise(tif_src)
tif_salt_pepper_noised_image = saltPepperNoise(tif_src, amount = 0.1)
tif_poisson_noised_image = poissonNoise(tif_src, factor=0.9)
tif_striping_noised_image = stripingNoise(tif_src, noise_strength= 70, direction='vertical')
tif_sun_angle_noised_image = sunAngleNoise(tif_src, intensity=1.0)
tif_vignetting_noised_image = vignettingNoise(tif_src)

# Denoise (TIF)
if tif_dem is None:
    tif_terrain_denoised_image = terrain(tif_terrain_noised_image)
else:
    tif_terrain_denoised_image = terrainTopo(
        tif_terrain_noised_image,
        DEM=tif_dem,
        sun_azimuth=225,
        sun_elevation=45,
        method="minnaert",
        mode="luminance",
        scale_clip=(0.8, 1.2),
        shadow_thresh=0.2,
        robust=True,
        ransac_iter=300,
        inlier_sigma=2.0
    )

tif_atmospheric_denoised_image = atmospher(tif_atmosphric_noised_image, factor=0.3, haze=True, rayleigh=True, yaml_name='KOMPSAT.yaml', sun_angle=30)
tif_gaussian_denoised_image = random(tif_gaussian_noised_image, type='gaussian')
tif_missing_denoised_image = missingLine(tif_missing_line_noised_image)
tif_salt_pepper_denoised_image = random(tif_salt_pepper_noised_image, type='saltPepper')
tif_poisson_denoised_image = random(tif_poisson_noised_image, type='poisson')
tif_striping_denoised_image = stripe(tif_striping_noised_image, direction='vertical')
tif_sun_angle_denoised_image = sunAngle(tif_sun_angle_noised_image)
tif_vignetting_denoised_image = vignetting(tif_vignetting_noised_image)

tif_denoised = {
    "Terrain": tif_terrain_denoised_image,
    "Atmospheric": tif_atmospheric_denoised_image,
    "Gaussian": tif_gaussian_denoised_image,
    "Missing Line": tif_missing_denoised_image,
    "Salt & Pepper": tif_salt_pepper_denoised_image,
    "Poisson": tif_poisson_denoised_image,
    "Striping": tif_striping_denoised_image,
    "Sun Angle": tif_sun_angle_denoised_image,
    "Vignetting": tif_vignetting_denoised_image
}

tif_noisy = {
    "Terrain": tif_terrain_noised_image,
    "Atmospheric": tif_atmosphric_noised_image,
    "Gaussian": tif_gaussian_noised_image,
    "Missing Line": tif_missing_line_noised_image,
    "Salt & Pepper": tif_salt_pepper_noised_image,
    "Poisson": tif_poisson_noised_image,
    "Striping": tif_striping_noised_image,
    "Sun Angle": tif_sun_angle_noised_image,
    "Vignetting": tif_vignetting_noised_image
}

# 결과 이미지 저장 (TIF prefix)
cv2.imwrite(os.path.join(noisy_dir, 'tif_terrain_noised_image.png'), tif_terrain_noised_image)
cv2.imwrite(os.path.join(noisy_dir, 'tif_atmosphric_noised_image.png'), tif_atmosphric_noised_image)
cv2.imwrite(os.path.join(noisy_dir, 'tif_gaussian_noised_image.png'), tif_gaussian_noised_image)
cv2.imwrite(os.path.join(noisy_dir, 'tif_missing_line_noised_image.png'), tif_missing_line_noised_image)
cv2.imwrite(os.path.join(noisy_dir, 'tif_salt_pepper_noised_image.png'), tif_salt_pepper_noised_image)
cv2.imwrite(os.path.join(noisy_dir, 'tif_poisson_noised_image.png'), tif_poisson_noised_image)
cv2.imwrite(os.path.join(noisy_dir, 'tif_striping_noised_image.png'), tif_striping_noised_image)
cv2.imwrite(os.path.join(noisy_dir, 'tif_sun_angle_noised_image.png'), tif_sun_angle_noised_image)
cv2.imwrite(os.path.join(noisy_dir, 'tif_vignetting_noised_image.png'), tif_vignetting_noised_image)

cv2.imwrite(os.path.join(denoised_dir, 'tif_terrain_denoised_image.png'), tif_terrain_denoised_image)
cv2.imwrite(os.path.join(denoised_dir, 'tif_atmospheric_denoised_image.png'), tif_atmospheric_denoised_image)
cv2.imwrite(os.path.join(denoised_dir, 'tif_gaussian_denoised_image.png'), tif_gaussian_denoised_image)
cv2.imwrite(os.path.join(denoised_dir, 'tif_missing_denoised_image.png'), tif_missing_denoised_image)
cv2.imwrite(os.path.join(denoised_dir, 'tif_salt_pepper_denoised_image.png'), tif_salt_pepper_denoised_image)
cv2.imwrite(os.path.join(denoised_dir, 'tif_poisson_denoised_image.png'), tif_poisson_denoised_image)
cv2.imwrite(os.path.join(denoised_dir, 'tif_striping_denoised_image.png'), tif_striping_denoised_image)
cv2.imwrite(os.path.join(denoised_dir, 'tif_sun_angle_denoised_image.png'), tif_sun_angle_denoised_image)
cv2.imwrite(os.path.join(denoised_dir, 'tif_vignetting_denoised_image.png'), tif_vignetting_denoised_image)

# TIF 평가지표 출력 및 저장
origin_tif_disp = dis(tif_src)
print("=== Similarity to Original TIF (Noisy images) ===")
for name, noisy_img in tif_noisy.items():
    m = compute_metrics(origin_tif_disp, noisy_img)
    print(f"[TIF] Noisy {name}: SSIM={m['SSIM']:.4f}, PSNR={m['PSNR']:.2f}, RMSE={m['RMSE']:.2f}, MAE={m['MAE']:.2f}")

print("=== Similarity to Original TIF (Denoised images) ===")
for name, denoised_img in tif_denoised.items():
    m = compute_metrics(origin_tif_disp, denoised_img)
    print(f"[TIF] Denoised {name}: SSIM={m['SSIM']:.4f}, PSNR={m['PSNR']:.2f}, RMSE={m['RMSE']:.2f}, MAE={m['MAE']:.2f}")

# CSV 요약 저장 (TIF)
tif_rows = []
tif_improvements = []
for name in tif_noisy.keys():
    nimg = tif_noisy[name]
    dimg = tif_denoised[name]

    m_on = compute_metrics(origin_tif_disp, nimg)
    m_od = compute_metrics(origin_tif_disp, dimg)
    m_nd = compute_metrics(nimg, dimg)

    tif_rows.append({
        "Noise": name, "Pair": "Original vs Noisy",
        "SSIM": m_on["SSIM"], "PSNR": m_on["PSNR"], "RMSE": m_on["RMSE"], "MAE": m_on["MAE"],
    })
    tif_rows.append({
        "Noise": name, "Pair": "Original vs Denoised",
        "SSIM": m_od["SSIM"], "PSNR": m_od["PSNR"], "RMSE": m_od["RMSE"], "MAE": m_od["MAE"],
    })
    tif_rows.append({
        "Noise": name, "Pair": "Noisy vs Denoised",
        "SSIM": m_nd["SSIM"], "PSNR": m_nd["PSNR"], "RMSE": m_nd["RMSE"], "MAE": m_nd["MAE"],
    })

    tif_improvements.append({
        "Noise": name,
        "dSSIM": m_od["SSIM"] - m_on["SSIM"],
        "dPSNR": m_od["PSNR"] - m_on["PSNR"],
        "dRMSE": m_on["RMSE"] - m_od["RMSE"],
        "dMAE": m_on["MAE"] - m_od["MAE"],
    })

pd.DataFrame(tif_rows).to_csv(os.path.join(csv_dir, 'metrics_summary_tif.csv'), index=False)
pd.DataFrame(tif_improvements).to_csv(os.path.join(csv_dir, 'metrics_improvement_tif.csv'), index=False)



# 결과 이미지 저장
cv2.imwrite(os.path.join(noisy_dir, 'terrain_noised_image.png'), terrain_noised_image)
cv2.imwrite(os.path.join(noisy_dir, 'atmosphric_noised_image.png'), atmosphric_noised_image)
cv2.imwrite(os.path.join(noisy_dir, 'gaussian_noised_image.png'), gaussian_noised_image)
cv2.imwrite(os.path.join(noisy_dir, 'missing_line_noised_image.png'), missing_line_noised_image)
cv2.imwrite(os.path.join(noisy_dir, 'salt_pepper_noised_image.png'), salt_pepper_noised_image)
cv2.imwrite(os.path.join(noisy_dir, 'poisson_noised_image.png'), poisson_noised_image)
cv2.imwrite(os.path.join(noisy_dir, 'striping_noised_image.png'), striping_noised_image)
cv2.imwrite(os.path.join(noisy_dir, 'sun_angle_noised_image.png'), sun_angle_noised_image)
cv2.imwrite(os.path.join(noisy_dir, 'vignetting_noised_image.png'), vignetting_noised_image)

cv2.imwrite(os.path.join(denoised_dir, 'terrain_denoised_image.png'), terrain_denoised_image)
cv2.imwrite(os.path.join(denoised_dir, 'atmospheric_denoised_image.png'), atmospheric_denoised_image)
cv2.imwrite(os.path.join(denoised_dir, 'gaussian_denoised_image.png'), gaussian_denoised_image)
cv2.imwrite(os.path.join(denoised_dir, 'missing_denoised_image.png'), missing_denoised_image)
cv2.imwrite(os.path.join(denoised_dir, 'salt_pepper_denoised_image.png'), salt_pepper_denoised_image)
cv2.imwrite(os.path.join(denoised_dir, 'poisson_denoised_image.png'), poisson_denoised_image)
cv2.imwrite(os.path.join(denoised_dir, 'striping_denoised_image.png'), striping_denoised_image)
cv2.imwrite(os.path.join(denoised_dir, 'sun_angle_denoised_image.png'), sun_angle_denoised_image)
cv2.imwrite(os.path.join(denoised_dir, 'vignetting_denoised_image.png'), vignetting_denoised_image)

# 결과 이미지 출력
'''
cv2.imshow('Terrain Noised Image', terrain_noised_image)
cv2.imshow('Atmosphric Noised Image', atmosphric_noised_image)
cv2.imshow('Gaussian Noised Image', gaussian_noised_image)
cv2.imshow('Missing Line Noised Image', missing_line_noised_image)
cv2.imshow('Salt Pepper Noised Image', salt_pepper_noised_image)
cv2.imshow('Poisson Noised Image', poisson_noised_image)
cv2.imshow('Striping Noised Image', striping_noised_image)
cv2.imshow('Sun Angle Noised Image', sun_angle_noised_image)
cv2.imshow('Vignetting Noised Image', vignetting_noised_image)
cv2.waitKey(0)
'''
"""
evaluate start
# evaluate (noiseEval param finder) — 비활성화
# terrain_param = evaluate(terrain_noised_image, 0.1, metric="rmse")
# atmospheric_param = evaluate(atmosphric_noised_image, 0.1, metric="rmse")
# gaussian_param = evaluate(gaussian_noised_image, 0.1, metric="rmse")
# missing_line_param = evaluate(missing_line_noised_image, 0.1, metric="rmse")
# salt_pepper_param = evaluate(salt_pepper_noised_image, 0.1, metric="rmse")
# poisson_param = evaluate(poisson_noised_image, 0.1, metric="rmse")
# striping_param = evaluate(striping_noised_image, 0.1, metric="rmse")
# sun_angle_param = evaluate(sun_angle_noised_image, 0.1, metric="rmse")
# vignetting_param = evaluate(vignetting_noised_image, 0.1, metric="rmse")

# CSV 파일로 저장 (비활성화)
# terrain_param.to_csv(os.path.join(csv_dir, 'terrain_param.csv'))
# atmospheric_param.to_csv(os.path.join(csv_dir, 'atmospheric_param.csv'))
# gaussian_param.to_csv(os.path.join(csv_dir, 'gaussian_param.csv'))
# missing_line_param.to_csv(os.path.join(csv_dir, 'missing_line_param.csv'))
# salt_pepper_param.to_csv(os.path.join(csv_dir, 'salt_pepper_param.csv'))
# poisson_param.to_csv(os.path.join(csv_dir, 'poisson_param.csv'))
# striping_param.to_csv(os.path.join(csv_dir, 'striping_param.csv'))
# sun_angle_param.to_csv(os.path.join(csv_dir, 'sun_angle_param.csv'))
# vignetting_param.to_csv(os.path.join(csv_dir, 'vignetting_param.csv'))
"""

#make_param_csv(src)
#make_param_csv(tif_src, save_path="tif_table.csv")