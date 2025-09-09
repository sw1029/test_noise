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

# 파라미터 타입 힌트: CSV에서 읽은 값들을 적절한 타입으로 캐스팅하기 위함
HINTS: Dict[str, Dict[str, Any]] = {
    'gaussian': {
        'mean': float, 'var': float,
    },
    'salt_pepper': {
        's_vs_p': float, 'amount': float,
    },
    'vignetting': {
        'strength': float, 'power': float,
    },
    'missingLine': {
        'num_threshold': int, 'len_threshold': int,
    },
    'striping': {
        'noise_strength': int, 'stripe_width': int, 'direction': str,
    },
    'sunAngle': {
        'angle': float, 'intensity': float, 'gamma': float,
    },
    'terrain': {
        'factor': float, 'slope': float, 'sun_angle': float,
        'sun_azimuth': float, 'sun_elevation': float,
        'Minnaert_constant_B': float, 'Minnaert_constant_G': float,
        'Minnaert_constant_R': float, 'Minnaert_constant_NIR': float,
    },
    'atmospheric': {
        'factor': float, 'haze': bool, 'rayleigh': bool, 'sun_angle': float,
    },
    'poisson': {
        'factor': float,
    },
}

def toBool(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    try:
        if isinstance(val, (int,)):
            return bool(val)
        if isinstance(val, float):
            return bool(int(val))
        s = str(val).strip().lower()
        if s in ("true", "t", "1", "yes", "y"): return True
        if s in ("false", "f", "0", "no", "n", ""): return False
    except Exception:
        pass
    return bool(val)

def casting(noiseType: str, params: Dict[str, Any]) -> Dict[str, Any]:
    hints = HINTS.get(noiseType, {})
    casted: Dict[str, Any] = {}
    for k, v in params.items():
        if v is None or pd.isna(v):
            continue
        typ = hints.get(k)
        try:
            if typ is int:
                casted[k] = int(v)
            elif typ is float:
                casted[k] = float(v)
            elif typ is bool:
                casted[k] = toBool(v)
            elif typ is str:
                casted[k] = str(v)
            else:
                # 힌트가 없으면 원본 유지
                casted[k] = v
        except Exception:
            # 캐스팅 실패 시 원본 값 유지
            casted[k] = v
    return casted

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

    # 타입 캐스팅 후 노이즈 적용
    params = casting(noiseType, params)
    noise = NOISE[noiseType]
    noisy_img = noise(src, **params)
    return noisy_img

__all__ = ['noiseGen']
