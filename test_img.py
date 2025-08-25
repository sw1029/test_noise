from test_noise.noise_generator import terrainNoise
from test_noise.noise_generator import atmosphericNoise
import cv2
import os

image_path = 'input_images/P0000__512__2304___1536.png'  # 원본 이미지 경로
output_dir = 'output_images'

# 출력 디렉토리 생성
os.makedirs(output_dir, exist_ok=True)

src = cv2.imread(image_path)

# Noise 추가
terrain_noised_image = terrainNoise(src, factor=0.8)
atmosphric_noised_image = atmosphericNoise(src, factor=0.8)

# 결과 이미지 저장
cv2.imwrite(os.path.join(output_dir, 'terrain_noised_image.png'), terrain_noised_image)
cv2.imwrite(os.path.join(output_dir, 'atmosphric_noised_image.png'), atmosphric_noised_image)

# 결과 이미지 출력
cv2.imshow('Terrain Noised Image', terrain_noised_image)
cv2.imshow('Atmosphric Noised Image', atmosphric_noised_image)
cv2.waitKey(0)

print("Noise addition completed and images saved.")