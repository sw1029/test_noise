from .gaussian_noise import GaussianNoise
from .poisson_noise import PoissonNoise
from .salt_pepper_noise import SaltPepperNoise
from .vignetting_noise import VignettingNoise
from .missing_line_noise import MissingLineNoise

gaussianNoise = GaussianNoise.add_noise
poissonNoise = PoissonNoise.add_noise
saltPepperNoise = SaltPepperNoise.add_noise
vignettingNoise = VignettingNoise.add_noise
missingLineNoise = MissingLineNoise.add_noise