'''
평가 프로세스 실행기

param finder를 통해 noise, metric별 파라미터를 찾고 dataframe 형태로 반환
'''

from .metric import *
from .param_finder import *
import pandas as pd
import numpy as np

NOISE_PARAM = [('gaussian', 'var'), 
               ('salt_pepper', 'amount'), 
               ('vignetting', 'strength'), 
               ('missingLine', 'num_threshold'), 
               ('striping', 'noise_strength'), 
               ('sunAngle', 'intensity'), 
               ('terrain', 'factor'), 
               ('atmospheric', 'factor'), 
               ('poisson', 'factor')]


def evaluate(src, target_value, metric="rmse", iter = 10, tol = 0.01) -> pd.DataFrame:
    params = {}
    for noise, param in NOISE_PARAM:
        param_value, value = find_param(src, target_value, noise_type=noise, 
                                 value_type=metric, iter=iter, tol=tol)
        params[noise] = {param: param_value , "val": value}
    return pd.DataFrame(params)