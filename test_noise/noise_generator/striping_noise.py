from base import Noise
import numpy as np

class StripingNoise(Noise):
    @staticmethod
    def add_noise(src, noise_strength=10, stripe_width=2, direction='horizontal') -> np.ndarray:
        rows, cols, channels = src.shape
        if direction == "horizontal":
            # 각 stripe_width 개의 행마다 동일한 노이즈 값을 가지도록 노이즈 생성
            num_stripes = rows // stripe_width  # stripe_width 줄씩 묶어서 노이즈를 적용할 수 있는 줄의 개수
            stripe_noise = np.random.randint(-noise_strength, noise_strength, size=(num_stripes, 1, 1))

            # stripe_width 줄씩 동일한 노이즈를 적용하여 새로운 노이즈 배열 생성
            stripe_noise = np.repeat(stripe_noise, stripe_width, axis=0)

            # 만약 총 행 수가 stripe_width로 나누어 떨어지지 않으면 남은 행에도 노이즈를 추가
            if stripe_noise.shape[0] < rows:
                extra_rows = rows - stripe_noise.shape[0]
                stripe_noise = np.vstack(
                    [stripe_noise, np.random.randint(-noise_strength, noise_strength, size=(extra_rows, 1, 1))])

            # 채널 수와 동일한 크기로 노이즈 확장 (BGR 모두 같은 노이즈 적용)
            stripe_noise = np.repeat(stripe_noise, channels, axis=2)

            # 원본 이미지에 노이즈 더하기
            striped_image_array = src + stripe_noise

        elif direction == "vertical":
            # 각 stripe_width 개의 열마다 동일한 노이즈 값을 가지도록 노이즈 생성
            num_stripes = cols // stripe_width
            stripe_noise = np.random.randint(-noise_strength, noise_strength, size=(1, num_stripes, 1))

            # stripe_width 열씩 동일한 노이즈를 적용하여 새로운 노이즈 배열 생성
            stripe_noise = np.repeat(stripe_noise, stripe_width, axis=1)

            # 만약 총 열 수가 stripe_width로 나누어 떨어지지 않으면 남은 열에도 노이즈를 추가
            if stripe_noise.shape[1] < cols:
                extra_cols = cols - stripe_noise.shape[1]
                stripe_noise = np.hstack(
                    [stripe_noise, np.random.randint(-noise_strength, noise_strength, size=(1, extra_cols, 1))])

            # 채널 수와 동일한 크기로 노이즈 확장 (BGR 모두 같은 노이즈 적용)
            stripe_noise = np.repeat(stripe_noise, channels, axis=2)

            # 원본 이미지에 노이즈 더하기
            striped_image_array = src + stripe_noise

        else:
            raise ValueError("direction 파라미터는 'horizontal' 또는 'vertical' 값만 허용합니다.")

        # 유효한 픽셀 값으로 클리핑
        striped_image_array = np.clip(striped_image_array, 0, 255)

        # 데이터 타입 변환 (uint8로 변환)
        striped_image = striped_image_array.astype(np.uint8)

        return striped_image