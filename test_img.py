from test_noise.noiseGenerator import *
from test_noise.noiseEval import *
from test_noise.denoise import *
import cv2
import os
import pandas as pd

image_path = 'input_images/P0000__512__2304___1536.png'  # 원본 이미지 경로
noisy_dir = 'output/noisy'
denoised_dir = 'output/denoised'
csv_dir = 'output/csv'

# 출력 디렉토리 생성
os.makedirs(noisy_dir, exist_ok=True)
os.makedirs(denoised_dir, exist_ok=True)
os.makedirs(csv_dir, exist_ok=True)

src = cv2.imread(image_path)

# Noise 추가
terrain_noised_image = terrainNoise(src, factor=0.3)
atmosphric_noised_image = atmosphericNoise(src, factor=0.3)
gaussian_noised_image = gaussianNoise(src)
missing_line_noised_image = missingLineNoise(src)
salt_pepper_noised_image = saltPepperNoise(src)
poisson_noised_image = poissonNoise(src)
striping_noised_image = stripingNoise(src)
sun_angle_noised_image = sunAngleNoise(src)
vignetting_noised_image = vignettingNoise(src)

# Denoise
terrain_denoised_image = terrain(terrain_noised_image, yaml_name='KOMPSAT.yaml', sun_angle=30)
atmospheric_denoised_image = atmospher(atmosphric_noised_image, haze=True, rayleigh=True, yaml_name='KOMPSAT.yaml', sun_angle=30)
gaussian_denoised_image = random(gaussian_noised_image, type='gaussian')
missing_denoised_image = missingLine(missing_line_noised_image)
salt_pepper_denoised_image = random(salt_pepper_noised_image, type='saltPepper')
poisson_denoised_image = random(poisson_noised_image, type='poisson')
striping_denoised_image = stripe(striping_noised_image)
sun_angle_denoised_image = sunAngle(sun_angle_noised_image)
vignetting_denoised_image = vignetting(vignetting_noised_image)


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
print("evaluate start")
# evaluate
terrain_param = evaluate(terrain_noised_image, 0.1, metric="rmse")
atmospheric_param = evaluate(atmosphric_noised_image, 0.1, metric="rmse")
gaussian_param = evaluate(gaussian_noised_image, 0.1, metric="rmse")
missing_line_param = evaluate(missing_line_noised_image, 0.1, metric="rmse")
salt_pepper_param = evaluate(salt_pepper_noised_image, 0.1, metric="rmse")
poisson_param = evaluate(poisson_noised_image, 0.1, metric="rmse")
striping_param = evaluate(striping_noised_image, 0.1, metric="rmse")
sun_angle_param = evaluate(sun_angle_noised_image, 0.1, metric="rmse")
vignetting_param = evaluate(vignetting_noised_image, 0.1, metric="rmse")

# CSV 파일로 저장
terrain_param.to_csv(os.path.join(csv_dir, 'terrain_param.csv'))
atmospheric_param.to_csv(os.path.join(csv_dir, 'atmospheric_param.csv'))
gaussian_param.to_csv(os.path.join(csv_dir, 'gaussian_param.csv'))
missing_line_param.to_csv(os.path.join(csv_dir, 'missing_line_param.csv'))
salt_pepper_param.to_csv(os.path.join(csv_dir, 'salt_pepper_param.csv'))
poisson_param.to_csv(os.path.join(csv_dir, 'poisson_param.csv'))
striping_param.to_csv(os.path.join(csv_dir, 'striping_param.csv'))
sun_angle_param.to_csv(os.path.join(csv_dir, 'sun_angle_param.csv'))
vignetting_param.to_csv(os.path.join(csv_dir, 'vignetting_param.csv'))

print("Noise addition completed and images saved.")