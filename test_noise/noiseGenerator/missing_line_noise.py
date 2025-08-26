from .base import NoiseBase
import numpy as np

class MissingLineNoise(NoiseBase):
    @staticmethod
    def add_noise(src, num_threshold=10, len_threshold=512) -> np.ndarray:
        rows, cols, channels = src.shape
        missing_line_image = src.copy()

        # 무작위로 삭제할 줄의 인덱스 선택
        missing_rows = np.random.choice(rows, size=np.random.randint(1, num_threshold + 1), replace=False)
        for row in missing_rows:
            # 결손 시작 위치와 길이 설정
            start_col = np.random.randint(0, cols / 2)  # 결손 시작 위치
            line_length = np.random.randint(100, len_threshold + 1)  # 결손 길이 (100부터 len_threshold까지)

            # 결손 구간이 이미지 경계를 넘어가지 않도록 설정
            end_col = min(start_col + line_length, cols)

            # 해당 가로 줄에서 결손 구간을 0으로 설정
            missing_line_image[row, start_col:end_col, :] = 0

        return missing_line_image