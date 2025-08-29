# Noise Generator 모듈

다양한 종류의 영상 노이즈를 생성하는 클래스를 포함하고 있습니다.

## 노이즈별 시각적 예시

샘플 이미지에 각 노이즈가 적용된 시각적 예시입니다.

### Atmospheric Noise
![Atmospheric Noise](../../output/noisy/atmosphric_noised_image.png)
Ls = H · ρ · T + Lp에서 H(total downwelling radiance)는 반사도-라디언스 변환식을 역으로 사용하고, T와 Lp는 Py6S로 계산해 선형 보간 형태로 구현합니다.

### Gaussian Noise
![Gaussian Noise](../../output/noisy/gaussian_noised_image.png)
이미지 전체에 정규 분포(가우시안 분포)를 따르는 랜덤 노이즈를 추가합니다.

### Missing Line Noise
![Missing Line Noise](../../output/noisy/missing_line_noised_image.png)
특정 행/열을 0으로 만들어 센서 라인 누락을 시뮬레이션합니다.

### Poisson Noise
![Poisson Noise](../../output/noisy/poisson_noised_image.png)
신호 강도에 따라 노이즈가 달라지는 푸아송(샷) 노이즈를 추가합니다.

### Salt & Pepper Noise
![Salt & Pepper Noise](../../output/noisy/salt_pepper_noised_image.png)
일부 픽셀을 0 또는 255로 치환하는 소금·후추 노이즈를 적용합니다.

### Striping Noise
![Striping Noise](../../output/noisy/striping_noised_image.png)
사인파 패턴을 더해 가로/세로 줄무늬를 만듭니다.

### Sun Angle Noise
![Sun Angle Noise](../../output/noisy/sun_angle_noised_image.png)
감마 보정과 태양 고도각을 이용해 전역 조도 변화를 시뮬레이션합니다.

### Terrain Noise
![Terrain Noise](../../output/noisy/terrain_noised_image.png)
Minnaert 역연산 기반으로 구현되며, DEM 입력 시 경사/입사각을 반영합니다.

### Vignetting Noise
![Vignetting Noise](../../output/noisy/vignetting_noised_image.png)
중앙부 대비 주변부가 어두워지는 비네팅 효과를 시뮬레이션합니다.
