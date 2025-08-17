from Py6S import *
# Create a SixS object
s = SixS()
# Set the wavelength to 0.675um
s.wavelength = Wavelength(0.675)
# Set the aerosol profile to Maritime
s.aero_profile = AeroProfile.PredefinedType(AeroProfile.Maritime)
# Run the model
s.run()
# Print some outputs
print(s.outputs.pixel_reflectance, s.outputs.pixel_radiance, s.outputs.direct_solar_irradiance)