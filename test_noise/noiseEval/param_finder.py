'''
균일한 노이즈 평가를 위한 파라미터 탐색기
'''

from ..noiseGenerator import *
from .metric import *
from ..utils import with_np_seed, make_eval_seeds
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
                progress: bool = False,
                eval_seeds: Optional[List[int]] = None,
                n_eval_seeds: int = 1,
                eval_seed_base: Optional[int] = None,
                return_details: bool = False,
                ) -> Tuple[Dict[str, Any], float]:
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

    # 고정된 평가 시드 집합 구성
    if eval_seeds is not None:
        seeds_for_eval: List[int] = list(map(int, eval_seeds))
    else:
        n = max(1, int(n_eval_seeds))
        base = int(eval_seed_base) if eval_seed_base is not None else (seed if seed is not None else 0)
        scope = f"{noise_type}|{value_type}"
        seeds_for_eval = make_eval_seeds(scope, base, n, ret_bits=31)

    def objective(trial: optuna.Trial) -> float:
        params = suggest(trial, noise_type)

        # 각 seed로 평가
        vals: List[float] = []
        for s in seeds_for_eval:
            with with_np_seed(int(s)):
                noisy_img = noise_fn(img, **params)
            v = float(metric_fn(img, noisy_img))
            vals.append(v)

        # 평균값을 목적함수로 사용
        mean_val = float(np.mean(vals)) if len(vals) > 0 else float('inf')
        err = abs(mean_val - target)

        # 결과 기록
        trial.set_user_attr('val', mean_val)
        trial.set_user_attr('vals', vals)
        trial.set_user_attr('seeds', seeds_for_eval)
        
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

    # 필요시 재평가
    best_vals: Optional[List[float]] = best.user_attrs.get('vals')
    best_seeds: Optional[List[int]] = best.user_attrs.get('seeds')

    if best_val is None or best_vals is None:
        # best trial에 기록이 없다면 동일 seed 집합으로 재평가
        recomputed_vals: List[float] = []
        for s in seeds_for_eval:
            with with_np_seed(int(s)):
                noisy_img = noise_fn(img, **best.params)
            recomputed_vals.append(float(metric_fn(img, noisy_img)))
        best_vals = recomputed_vals
        best_seeds = seeds_for_eval
        best_val = float(np.mean(best_vals)) if len(best_vals) > 0 else float('inf')

    details: Dict[str, Any] = {
        'eval_seeds': list(map(int, best_seeds)) if best_seeds is not None else list(map(int, seeds_for_eval)),
        'per_seed_vals': list(map(float, best_vals)) if best_vals is not None else [],
    }
    try:
        arr = np.array(details['per_seed_vals'], dtype=float)
        details['val_mean'] = float(np.mean(arr)) if arr.size else float('inf')
        details['val_std'] = float(np.std(arr)) if arr.size else float('nan')
    except Exception:
        details['val_mean'] = float(best_val) if best_val is not None else float('inf')
        details['val_std'] = float('nan')

    # 요청에 따라 details 포함 또는 기존 반환 형태 유지
    if return_details:
        return best.params, float(best_val), details
    else:
        return best.params, float(best_val)


# 단일 탐색 전용
def find_param(img, target, noise_type='gaussian', value_type="rmse", iter=100, tol=0.01):
    params, val = find_params(img, target, noise_type=noise_type, value_type=value_type, n_trials=iter, tol=tol)
    main_param = None
    specs = PARAMS.get(noise_type, [])
    if specs:
        main_param = params.get(specs[0]['name'])
    return (main_param if main_param is not None else 0.0), val
