'''
균일한 노이즈 평가를 위한 파라미터 탐색기
'''

from ..noiseGenerator import *
from .metric import *
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import optuna

NOISE = {
    'gaussian': gaussianNoise,
    'salt_pepper': saltPepperNoise,
    'vignetting': vignettingNoise,
    'missingLine': missingLineNoise,
    'striping': stripingNoise,
    'sunAngle': sunAngleNoise,
    'terrain': terrainNoise,
    'atmospheric': atmosphericNoise,
    'poisson': poissonNoise
}

# metric별 탐색을 위한 탐색공간 정의
# positive는 증가/감소에 따른 개선여부
METRIC = {
    'psnr': {'func': psnr, 'positive': False, 'start': 22.0, 'end': 30.0, 'step': 1.0},
    'ssim': {'func': ssim, 'positive': False, 'start': 0.6,  'end': 0.95, 'step': 0.05},
    'rmse': {'func': rmse, 'positive': True,  'start': 3.0,  'end': 20.0, 'step': 1.0},
    'mae':  {'func': mae,  'positive': True,  'start': 1.0,  'end': 8.0,  'step': 1.0},
}

# 노이즈별 탐색 파라미터 공간 정의
PARAMS: Dict[str, List[Dict[str, Any]]] = {
    'gaussian': [
        { 'name': 'mean', 'type': 'float', 'low': -50.0, 'high': 50.0 },
        { 'name': 'var',  'type': 'float', 'low': 1.0, 'high': 5000.0, 'log': True },
    ],
    'salt_pepper': [
        { 'name': 's_vs_p', 'type': 'float', 'low': 0.0, 'high': 1.0 },
        { 'name': 'amount', 'type': 'float', 'low': 0.0, 'high': 0.5 },
    ],
    'vignetting': [
        { 'name': 'strength', 'type': 'float', 'low': 0.0, 'high': 1.0 },
        { 'name': 'power',    'type': 'float', 'low': 1.0, 'high': 5.0 },
    ],
    'missingLine': [
        { 'name': 'num_threshold', 'type': 'int', 'low': 1, 'high': 50 },
        { 'name': 'len_threshold', 'type': 'int', 'low': 50, 'high': 1024 },
    ],
    'striping': [
        { 'name': 'noise_strength', 'type': 'int', 'low': 1, 'high': 50 },
        { 'name': 'stripe_width',   'type': 'int', 'low': 1, 'high': 16 },
        { 'name': 'direction',      'type': 'categorical', 'choices': ['horizontal', 'vertical'] },
    ],
    'sunAngle': [
        { 'name': 'angle',     'type': 'float', 'low': 5.0,  'high': 85.0 },
        { 'name': 'intensity', 'type': 'float', 'low': 0.01, 'high': 1.0 },
        { 'name': 'gamma',     'type': 'float', 'low': 0.8,  'high': 3.0 },
    ],
    'terrain': [
        { 'name': 'factor',   'type': 'float', 'low': 0.0,  'high': 1.0 },
        { 'name': 'slope',    'type': 'float', 'low': 0.0,  'high': 45.0 },
        { 'name': 'sun_angle','type': 'float', 'low': 10.0, 'high': 80.0 },
        { 'name': 'sun_azimuth','type': 'float', 'low': 0.0, 'high': 360.0 },
        { 'name': 'sun_elevation','type': 'float', 'low': 10.0, 'high': 80.0 },
        { 'name': 'Minnaert_constant_B','type': 'float', 'low': 0.1, 'high': 1.0 },
        { 'name': 'Minnaert_constant_G','type': 'float', 'low': 0.1, 'high': 1.0 },
        { 'name': 'Minnaert_constant_R','type': 'float', 'low': 0.1, 'high': 1.0 },
        { 'name': 'Minnaert_constant_NIR','type': 'float','low': 0.1, 'high': 1.0 },
    ],
    'atmospheric': [
        { 'name': 'factor',   'type': 'float', 'low': 0.0, 'high': 1.0 },
        { 'name': 'haze',     'type': 'bool' },
        { 'name': 'rayleigh', 'type': 'bool' },
        { 'name': 'sun_angle','type': 'float', 'low': 5.0, 'high': 85.0 },
    ],
    'poisson': [
        { 'name': 'factor', 'type': 'float', 'low': 0.0, 'high': 1.0 },
    ],
}


def suggest(trial: optuna.Trial, noise_type: str) -> Dict[str, Any]:
    specs = PARAMS.get(noise_type)
    if specs is None:
        return {}
    params: Dict[str, Any] = {}
    for p in specs:
        p_type = p.get('type', 'float')
        name = p['name']
        if p_type == 'float':
            low = float(p['low']); high = float(p['high'])
            step = p.get('step')
            log = bool(p.get('log', False))
            if step is not None:
                params[name] = trial.suggest_float(name, low, high, step=step, log=log)
            else:
                params[name] = trial.suggest_float(name, low, high, log=log)
        elif p_type == 'int':
            low = int(p['low']); high = int(p['high'])
            step = int(p.get('step', 1))
            params[name] = trial.suggest_int(name, low, high, step=step)
        elif p_type == 'categorical':
            choices = p['choices']
            params[name] = trial.suggest_categorical(name, choices)
        elif p_type == 'bool':
            params[name] = trial.suggest_categorical(name, [True, False])
        else:
            continue
    return params


def find_params(img: np.ndarray,
                target: float,
                noise_type: str = 'gaussian',
                value_type: str = 'rmse',
                n_trials: int = 100,
                tol: float = 0.01,
                seed: Optional[int] = None,
                quiet: bool = True,
                progress: bool = False) -> Tuple[Dict[str, Any], float]:
    """
    Optuna를 사용하여 지정한 metric 값(target)에 가장 근접한 파라미터 조합을 탐색.
    """

    noise_fn = NOISE[noise_type]
    metric_fn = METRIC[value_type]['func']
    target = float(target)

    # Optuna 로그 억제
    if quiet:
        try:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except Exception:
            pass

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction='minimize', sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        params = suggest(trial, noise_type)
        noisy_img = noise_fn(img, **params)
        val = float(metric_fn(img, noisy_img))
        err = abs(val - target)
        # 결과 기록
        trial.set_user_attr('val', val)
        
        if err <= tol:
            # 목표 오차 이내에 도달한 경우 조기종료
            try:
                study.stop()
            except Exception:
                pass
            return 0.0  # 최소값으로 종료조건 return
        return err

    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=bool(progress))
    
    best = study.best_trial
    best_val = best.user_attrs.get('val', None)
    if best_val is None:
        noisy_img = noise_fn(img, **best.params)
        best_val = float(metric_fn(img, noisy_img))

    return best.params, float(best_val)


# 단일 탐색 전용
def find_param(img, target, noise_type='gaussian', value_type="rmse", iter=100, tol=0.01):
    params, val = find_params(img, target, noise_type=noise_type, value_type=value_type, n_trials=iter, tol=tol)
    main_param = None
    specs = PARAMS.get(noise_type, [])
    if specs:
        main_param = params.get(specs[0]['name'])
    return (main_param if main_param is not None else 0.0), val
