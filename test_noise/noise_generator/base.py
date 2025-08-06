from abc import ABC, abstractmethod
import numpy as np

class Noise(ABC):
    """
    모든 노이즈에 대한 기본 클래스입니다. 추후 기능 확장시 해당 클래스를 상속하여 확장합니다.

    factor, threshold 등의 파라미터를 추가 시 init을 정의 후 입력받게 하여 구현 가능합니다.
    """
    @staticmethod
    @abstractmethod
    def add_noise(src: np.ndarray, **kwargs) -> np.ndarray:
        """
        입력된 이미지에 노이즈를 적용하는 부분입니다.
        init에서 입력받은 이미지에 noise를 추가한 후 반환합니다. 
        """
        pass
