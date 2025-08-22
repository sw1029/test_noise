'''
균일한 노이즈 평가를 위한 파라미터 탐색기
'''

from noise_generator import *
from metric import *

def find_param(img, target_value, noise_type = 'gaussian',value_type = "rmse"):
    '''
    이진 탐색을 통하여 선택한 noise의 target_value에 도달하는 파라미터를 찾는다.
    '''
    pass