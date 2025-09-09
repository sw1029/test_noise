from .base import NoiseBase
import numpy as np

class MissingLineNoise(NoiseBase):
    @staticmethod
    def add_noise(src, num_threshold=10, len_threshold=512) -> np.ndarray:
        rows, cols, channels = src.shape
        missing_line_image = src.copy()

        max_pick = max(1, min(rows, num_threshold))
        pick_size = np.random.randint(1, max_pick + 1)
        missing_rows = np.random.choice(rows, size=pick_size, replace=False)
        for row in missing_rows:
            # 결손 시작 위치와 길이 설정
            start_high = max(1, cols // 2)
            start_col = np.random.randint(0, start_high)  # 결손 시작 위치

            # 결손 길이 샘플링을 항상 유효 범위로 보정
            max_len = max(1, cols - start_col)
            hi_candidate = min(int(len_threshold), max_len)
            lo = 100
            if hi_candidate <= lo:
                line_length = hi_candidate
            else:
                line_length = np.random.randint(lo, hi_candidate + 1) 

            # 결손 구간이 이미지 경계를 넘어가지 않도록 설정
            end_col = min(start_col + line_length, cols)

            # 해당 가로 줄에서 결손 구간을 0으로 설정
            missing_line_image[row, start_col:end_col, :] = 0

        return missing_line_image