from .denoiseAtmospheric import DenoiseAtmospheric
from .denoiseRandom import DenoiseRandom
from .denoiseMissingLine import DenoiseMissingLine
from .denoiseStripe import DenoiseStripe
from .denoiseSunAngle import DenoiseSunAngle
from .denoiseVignetting import DenoiseVignetting
from .denoiseTerrain import DenoiseTerrain

atmospher = DenoiseAtmospheric.denoise
random = DenoiseRandom.denoise
missingLine = DenoiseMissingLine.denoise
stripe = DenoiseStripe.denoise_algotom
sunAngle = DenoiseSunAngle.denoise
vignetting = DenoiseVignetting.denoise
terrain = DenoiseTerrain.denoise