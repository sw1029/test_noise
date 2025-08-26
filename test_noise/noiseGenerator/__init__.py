from .gaussian_noise import GaussianNoise
from .poisson_noise import PoissonNoise
from .salt_pepper_noise import SaltPepperNoise
from .vignetting_noise import VignettingNoise
from .missing_line_noise import MissingLineNoise
from .atmospheric_noise import AtmosphericNoise
from .terrain_noise import TerrainNoise
from .sun_angle_noise import SunAngleNoise
from .striping_noise import StripingNoise

gaussianNoise = GaussianNoise.add_noise
poissonNoise = PoissonNoise.add_noise
saltPepperNoise = SaltPepperNoise.add_noise
vignettingNoise = VignettingNoise.add_noise
missingLineNoise = MissingLineNoise.add_noise
atmosphericNoise = AtmosphericNoise.add_noise
terrainNoise = TerrainNoise.add_noise
sunAngleNoise = SunAngleNoise.add_noise
stripingNoise = StripingNoise.add_noise