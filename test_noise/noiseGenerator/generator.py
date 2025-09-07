import pandas as pd
from typing import Dict, Any

from .gaussian_noise import GaussianNoise
from .salt_pepper_noise import SaltPepperNoise
from .vignetting_noise import VignettingNoise
from .missing_line_noise import MissingLineNoise
from .striping_noise import StripingNoise
from .sun_angle_noise import SunAngleNoise
from .terrain_noise import TerrainNoise
from .atmospheric_noise import AtmosphericNoise
from .poisson_noise import PoissonNoise

NOISE: Dict[str, Any] = {
    'gaussian': GaussianNoise.add_noise,
    'salt_pepper': SaltPepperNoise.add_noise,
    'vignetting': VignettingNoise.add_noise,
    'missingLine': MissingLineNoise.add_noise,
    'striping': StripingNoise.add_noise,
    'sunAngle': SunAngleNoise.add_noise,
    'terrain': TerrainNoise.add_noise,
    'atmospheric': AtmosphericNoise.add_noise,
    'poisson': PoissonNoise.add_noise,
}

def noiseGen(src, tablePath, noiseType, metricType, targetValue, tol=0.1):
    '''
    make_param_csv를 통해 생성된 csv를 dataframe 형태로 로드하여 사용
    noiseType에 따른 noise를 noiseGenerator를 통해 호출하여 생성 후 반환하되, 파라미터는 테이블에서 불러옴
    metric 은 noiseEval의 metric을 통해 가져옴
    target value에 가장 근접한 민감도를 mertic type, noise type에 맞게 가져오되, tol 이내의 오차범위 내의 파라미터 조합을 탐색할 수 없는 경우 에러 메시지를 반환
    '''
    df = pd.read_csv(tablePath)
    cand = df[(df['noise_type'] == noiseType) & (df['metric_type'] == metricType)].copy()
    if cand.empty:
        raise ValueError("해당 noise/metric 조합의 파라미터가 테이블에 없습니다.")
    cand['diff'] = (cand['value'] - float(targetValue)).abs()
    best = cand.sort_values('diff', ascending=True).head(1)
    if best.empty or best.iloc[0]['diff'] > tol:
        raise ValueError("요청한 오차 범위 내의 파라미터를 찾지 못했습니다.")

    # 파라미터 추출
    params = {}
    for col in best.columns:
        if col.startswith('param_'):
            name = col[len('param_'):]
            val = best.iloc[0][col]
            # NaN은 건너뜀
            if pd.notna(val):
                params[name] = val

    # 노이즈 적용
    noise = NOISE[noiseType]
    noisy_img = noise(src, **params)
    return noisy_img

__all__ = ['noiseGen']
