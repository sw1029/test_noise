# Denoise 모듈

다양한 종류의 영상 노이즈를 제거하는 클래스를 포함하고 있습니다.

## 노이즈 제거 시각적 예시

각 노이즈가 제거된 후의 시각적 예시입니다.

### Atmospheric Denoise
`Py6S` 라이브러리를 사용하여 대기 중의 헤이즈(haze) 및 레일리(Rayleigh) 산란 효과를 역으로 적용하여 제거합니다. 기존 noise 적용 연산의 역연산으로 이루어집니다.
![Atmospheric Denoise](../../../output/denoised/atmospheric_denoised_image.png)

### Gaussian Denoise
`cv2.GaussianBlur`를 이용한 간단한 가우시안 블러로 노이즈를 제거합니다.
![Gaussian Denoise](../../../output/denoised/gaussian_denoised_image.png)

### Missing Line Denoise
`skimage`의 `inpaint_biharmonic`을 사용하여 누락된 라인(0으로 처리된 픽셀)을 주변 픽셀을 기반으로 채워 넣습니다.
![Missing Line Denoise](../../../output/denoised/missing_denoised_image.png)

### Poisson Denoise
`cv2.medianBlur`를 이용한 미디언 필터링으로 노이즈를 제거합니다.
![Poisson Denoise](../../../output/denoised/poisson_denoised_image.png)

### Salt & Pepper Denoise
`cv2.medianBlur`를 이용한 median 필터링으로 노이즈를 제거합니다.
![Salt & Pepper Denoise](../../../output/denoised/salt_pepper_denoised_image.png)

### Striping Denoise
오픈소스 algotom을 이용하여 denoise를 수행합니다. 푸리에 변환(FFT)을 기반으로 주파수 영역에서 주기적인 줄무늬 노이즈를 탐지하고 억제하여 제거하는 기능 역시 구현되어 있습니다.  
![Striping Denoise](../../../output/denoised/striping_denoised_image.png)

### Sun Angle Denoise
태양 고도각에 따른 조도 변화를 시뮬레이션한 노이즈를 간단한 absolute correciton으로 제거합니다.
![Sun Angle Denoise](../../../output/denoised/sun_angle_denoised_image.png)

### Terrain Denoise
Minnaert 연산을 통해 보정을 수행합니다. DEM이 주어진 경우 DEM을 바탕으로 noise 생성의 역연산을 통해 보정을 수행합니다.
![Terrain Denoise](../../../output/denoised/terrain_denoised_image.png)

### Vignetting Denoise
비네팅 마스크를 계산하여 이미지에 나누는 방식으로 가장자리 어둡기 효과를 보정합니다.
![Vignetting Denoise](../../../output/denoised/vignetting_denoised_image.png)
