from .gaussian_noise import GaussianNoise
from .poisson_noise import PoissonNoise
from .salt_pepper_noise import SaltPepperNoise
from .vignetting_noise import VignettingNoise
from .missing_line import MissingLineNoise

gaussian_noise = GaussianNoise.add_noise
poisson_noise = PoissonNoise.add_noise
salt_pepper_noise = SaltPepperNoise.add_noise
vignetting_noise = VignettingNoise.add_noise
missing_line_noise = MissingLineNoise.add_noise