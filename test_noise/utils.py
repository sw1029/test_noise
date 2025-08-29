import math
import numpy as np
from Py6S import SixS, AtmosProfile, AeroProfile, GroundReflectance, Wavelength, Geometry
import cv2

# radiance <-> reflectance
def radiance2reflectance(radiance, distance, ESUN, solar_angle):
    return (math.pi) * radiance * (distance ** 2) / (ESUN * math.cos(solar_angle))

def reflectance2radiance(reflectance, distance, ESUN, solar_angle):
    return reflectance * ESUN * math.cos(solar_angle) / (math.pi * (distance ** 2))


# radiance <-> DN
def DN2radiance(DN,gain,offset):
    return gain * DN + offset
def radiance2DN(radiance, gain, offset):
    return (radiance - offset) / gain


# Minnaert correction 역연산
def inverse_Minnaert(radiance,i,e,k):
    '''
    i : 태양 입사각
    e : slope
    k : Minnaert 상수

    return : 역연산을 적용한 radiance
    '''
    return radiance / (np.cos(np.deg2rad(e)) ** (k-1) * np.cos(np.deg2rad(i)) ** k)

def Minnaert(radiance, i,e,k):
    '''
    i : 태양 입사각
    e : slope
    k : Minnaert 상수

    return : Minnaert 보정값
    '''
    return radiance * (np.cos(np.deg2rad(e)) ** (k-1)) * (np.cos(np.deg2rad(i)) ** k)

def get_rad0_rad1(wavelength, solar_zenith, haze=True):
    s = SixS()
    s.altitudes.set_sensor_satellite_level()
    s.altitudes.set_target_sea_level()
    s.atmos_profile = AtmosProfile.PredefinedType(AtmosProfile.MidlatitudeSummer)
    if haze:
        s.aero_profile = AeroProfile.PredefinedType(AeroProfile.Continental)
    else:
        s.aero_profile = AeroProfile.PredefinedType(AeroProfile.NoAerosols)

    s.wavelength = Wavelength(wavelength) #

    s.geometry = Geometry.User()
    s.geometry.solar_z = np.rad2deg(solar_zenith)
    s.geometry.solar_a = 0
    s.geometry.view_z = 0
    s.geometry.view_a = 0
    s.ground_reflectance = GroundReflectance.HomogeneousLambertian(0.0)
    s.run()
    rad0 = s.outputs.pixel_radiance
    s.ground_reflectance = GroundReflectance.HomogeneousLambertian(1.0)
    s.run()
    rad1 = s.outputs.pixel_radiance
    return rad0, rad1

def dis(image):
    """
    float -> uint8 이미지 정규화
    """
    if image.ndim == 2:
        img_copy = image.astype(np.float32)
        if np.max(img_copy) > np.min(img_copy):
            img_copy = (img_copy - np.min(img_copy)) / (np.max(img_copy) - np.min(img_copy))
        return (img_copy * 255).astype(np.uint8)
    elif image.ndim == 3 and image.shape[2] == 3:
        normalized_channels = []
        for i in range(3):
            channel = image[:, :, i].astype(np.float32)
            if np.max(channel) > np.min(channel):
                channel = (channel - np.min(channel)) / (np.max(channel) - np.min(channel))
            else:
                channel.fill(0)
            normalized_channels.append(channel * 255)
        display_image = cv2.merge(normalized_channels)
        return display_image.astype(np.uint8) 
    else:
        try:
            return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        except cv2.error:
            return np.zeros(image.shape[:2], dtype=np.uint8)