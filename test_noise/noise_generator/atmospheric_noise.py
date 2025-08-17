from base import Noise
import numpy as np

class AtmosphericNoise(Noise):
    @staticmethod
    def add_noise(src,haze=True,raylaigh=True) -> np.ndarray:
       pass