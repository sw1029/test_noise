'''
평가 프로세스 실행기

param finder를 통해 noise, metric별 파라미터를 찾고 dataframe 형태로 반환
'''

from .metric import *
from .param_finder import *
from ..utils import make_eval_seeds
from tqdm import tqdm
import pandas as pd
import numpy as np
from typing import Optional

# (noise type, [(paramName, low, high), ...]) 형태로 구성
NOISE_PARAM = [
    (k, [(p['name'], p.get('low', None), p.get('high', None))
         if p.get('type') in (None, 'int', 'float') else (p['name'], None, None)
         for p in v])
    for k, v in PARAMS.items()
]


def evaluate(src, target_value, metric="rmse", iter=50, tol=0.01) -> pd.DataFrame:
    '''
    각 noise 타입별로 target_value에 가장 근접한 파라미터 조합을 탐색하여 DataFrame 형태로 반환.
    반환 DataFrame은 각 noise를 column으로, 행에는 파라미터명과 'val'(달성값)을 기록.
    '''
    params = {}
    for noise, _ in NOISE_PARAM:
        found_params, value = find_params(src, target_value,
                                          noise_type=noise, value_type=metric,
                                          n_trials=iter, tol=tol)
        params[noise] = {**found_params, "val": value}
    return pd.DataFrame(params)


def make_param_csv(src,
                   iter=500,
                   tol=0.01,
                   save_path="table.csv",
                   early_stop: bool = True,
                   patience: int = 32,
                   val_eps: Optional[float] = None,
                   verbose: bool = False,
                   optuna_quiet: bool = True,
                   optuna_progress: bool = False,
                   optuna_seed: Optional[int] = 0,
                   seeds: int = 1,
                   eval_seed_base: Optional[int] = 12345,
                   store_per_seed_values: bool = True):
    '''
    각 metric 별로 0.1 단위로 분할하여 모든 민감도 수치를 재현하기 위한 파라미터 테이블을 csv 형태로 저장
    프로젝트 루트 디렉토리에 table.csv 형태로 저장
    noiseEval의 param_finder와 evaluator를 활용하는 형태
    '''
    rows = []
    for noise_type, _ in NOISE_PARAM:
        for metric_type in METRIC.keys():
            grid = METRIC.get(metric_type)
            if grid is None:
                continue
            targets = np.arange(grid['start'], grid['end'] + 1e-9, grid['step'])
            last_val = None
            stagnant = 0
            eps = val_eps if val_eps is not None else max(float(grid.get('step', 0.0) or 0.0) * 0.1, float(tol) * 0.5)

            # 재현성을 위한 n개 시드
            if seeds and seeds > 1:
                base = int(eval_seed_base) if eval_seed_base is not None else 0
                scope = f"{noise_type}|{metric_type}"
                eval_seeds = make_eval_seeds(scope, base, int(seeds), ret_bits=31)
            else:
                eval_seeds = None

            for target in tqdm(targets, desc=f"{noise_type}:{metric_type}"):
                try:
                    if eval_seeds is None and (seeds is None or int(seeds) <= 1):
                        # 단일 평가
                        params, val = find_params(src, float(target),
                                                  noise_type=noise_type,
                                                  value_type=metric_type,
                                                  n_trials=iter, tol=tol,
                                                  seed=optuna_seed,
                                                  quiet=optuna_quiet,
                                                  progress=optuna_progress)
                        details = None
                    else:
                        # 다중 시드 평균 평가 + 시드/값 기록
                        params, val, details = find_params(src, float(target),
                                                           noise_type=noise_type,
                                                           value_type=metric_type,
                                                           n_trials=iter, tol=tol,
                                                           seed=optuna_seed,
                                                           quiet=optuna_quiet,
                                                           progress=optuna_progress,
                                                           eval_seeds=eval_seeds,
                                                           n_eval_seeds=int(seeds),
                                                           eval_seed_base=int(eval_seed_base) if eval_seed_base is not None else None,
                                                           return_details=True)
                except Exception:
                    params, val = {}, float('inf')
                    details = None

                row = {
                    'noise_type': noise_type,
                    'metric_type': metric_type,
                    'target': float(target),
                    # value는 평균값으로 기록
                    'value': float(val) if val is not None else np.nan,
                    # 보조 지표: 표준편차 및 시드 기록
                    'value_std': (float(details.get('val_std')) if details and (details.get('val_std') is not None) else np.nan),
                    'eval_seeds': (
                        ','.join(map(str, details.get('eval_seeds'))) if (details and details.get('eval_seeds')) else (
                            ','.join(map(str, eval_seeds)) if eval_seeds else ''
                        )
                    ),
                    'per_seed_values': (
                        ','.join(map(lambda x: f"{float(x):.6g}", details.get('per_seed_vals'))) if (details and details.get('per_seed_vals')) else ''
                    ),
                    'success': (val is not None) and (np.isfinite(val)) and (abs(val - float(target)) <= tol)
                }
                for k, v in params.items():
                    row[f'param_{k}'] = v
                rows.append(row)

                # Early stopping
                if val is None or not np.isfinite(val):
                    stagnant += 1
                else:
                    # 목표에 도달했으면 정체 카운터 리셋
                    if abs(float(val) - float(target)) <= float(tol):
                        stagnant = 0
                    else:
                        if last_val is not None and abs(float(val) - float(last_val)) <= float(eps):
                            stagnant += 1
                        else:
                            stagnant = 0

                last_val = float(val) if val is not None and np.isfinite(val) else last_val

                if early_stop and stagnant >= int(patience):
                    if verbose:
                        print(f"[EarlyStop] {noise_type}/{metric_type}: break at target={float(target):.6g} after {stagnant} stagnant steps (eps={eps}).")
                    break
            pd.DataFrame(rows).to_csv(save_path, index=False) # 중간 저장
    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    return df
