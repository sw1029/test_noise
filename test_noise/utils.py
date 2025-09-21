import numpy as np
import math
from Py6S import SixS, AtmosProfile, AeroProfile, GroundReflectance, Wavelength, Geometry
import cv2
import os
import random
import hashlib
import contextlib
from typing import List, Optional

# DEM을 통해 slope와 태양 입사각 계산
def angle(DEM, sun_azimuth, sun_elevation, pixel_size=1.0):
    """
    DEM의 기울기(slope)와 태양 입사각(incidence angle)을 계산합니다.
    """
    sun_azimuth_rad = np.deg2rad(sun_azimuth)
    sun_elevation_rad = np.deg2rad(sun_elevation)

    # dy: 행(세로, 북-남) 방향 픽셀 간격, dx: 열(가로, 서-동) 방향 픽셀 간격
    # 단위는 DEM의 수평 좌표 단위. 스칼라 입력 시 dy=dx=pixel_size로 사용
    if isinstance(pixel_size, (tuple, list, np.ndarray)) and len(pixel_size) == 2:
        dy, dx = float(pixel_size[0]), float(pixel_size[1])
    else:
        dy = dx = float(pixel_size)

    dz_dy, dz_dx = np.gradient(DEM, dy, dx)

    slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    slope_deg = np.rad2deg(slope_rad)

    x = -dz_dx
    y = -dz_dy
    z = np.ones_like(DEM)

    magnitude = np.sqrt(x**2 + y**2 + z**2)
    magnitude = np.maximum(magnitude, 1e-6)
    x /= magnitude
    y /= magnitude
    z /= magnitude

    zenith_rad = np.deg2rad(90 - sun_elevation)
    sun_x = np.sin(zenith_rad) * np.sin(sun_azimuth_rad)
    sun_y = np.sin(zenith_rad) * np.cos(sun_azimuth_rad)
    sun_z = np.cos(zenith_rad)

    cos_i = x * sun_x + y * sun_y + z * sun_z
    cos_i = np.clip(cos_i, -1.0, 1.0)
    angle_deg = np.rad2deg(np.arccos(cos_i))

    return angle_deg, slope_deg

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
    cos_i_val = np.cos(np.deg2rad(i))
    cos_e_val = np.cos(np.deg2rad(e))

    cos_i_val = np.maximum(cos_i_val, 0)
    cos_e_val = np.maximum(cos_e_val, 0)

    cos_i_val = np.maximum(cos_i_val, 1e-6)
    cos_e_val = np.maximum(cos_e_val, 1e-6)
    
    return radiance / (cos_e_val ** (k-1) * cos_i_val ** k)

def Minnaert(radiance, i,e,k):
    '''
    i : 태양 입사각
    e : slope
    k : Minnaert 상수

    return : Minnaert 보정값
    '''
    cos_i_val = np.cos(np.deg2rad(i))
    cos_e_val = np.cos(np.deg2rad(e))

    cos_i_val = np.maximum(cos_i_val, 0)
    cos_e_val = np.maximum(cos_e_val, 0)
    
    return radiance * (cos_e_val ** (k-1)) * (cos_i_val ** k)

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
                channel.fill(0.01)
            normalized_channels.append(channel * 255)
        display_image = cv2.merge(normalized_channels)
        return display_image.astype(np.uint8) 
    else:
        try:
            return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        except cv2.error:
            return np.zeros(image.shape[:2], dtype=np.uint8)


# 시드 관련 기능들
# 동일한 입력에서 항상 동일한 시드/결과를 얻기 위함
# 파이썬 내장 hash는 세션마다 달라질 수 있어, 안정 해시(SHA-256) 기반으로 고정성 확보
# 무작위 시드만 사용하면 실행마다 값이 달라져 비교/디버깅이 어려울 수 있음
def seed_all(seed: int, set_env: bool = False) -> None:
    """
    재현성을 위해 전역 시드를 설정
    필요 시 자식 프로세스를 위해 환경변수 PYTHONHASHSEED 설정
    """
    s = int(seed)
    try:
        np.random.seed(s)
    except Exception:
        pass
    try:
        random.seed(s)
    except Exception:
        pass
    if set_env:
        os.environ['PYTHONHASHSEED'] = str(s)


@contextlib.contextmanager
def with_np_seed(seed: int):
    """
    numpy 전역 난수 시드를 일시적으로 설정하고 사용 후 원상 복구
    """
    state = np.random.get_state()
    try:
        np.random.seed(int(seed))
        yield
    finally:
        try:
            np.random.set_state(state)
        except Exception:
            pass


def hash_to_int(text: str, bits: int = 64) -> int:
    """
    SHA-256을 이용해 플랫폼에 독립적인 안정 해시를 정수로 변환
    bits는 결과 정수의 비트 폭(<= 256)을 의미
    """
    h = hashlib.sha256(text.encode('utf-8')).digest()
    val = int.from_bytes(h, 'big')
    if bits <= 0 or bits > 256:
        bits = 64
    mod = (1 << bits)
    return val % mod


def make_eval_seeds(scope: str, base_seed: int, n: int, ret_bits: int = 31) -> List[int]:
    """
    안정 해시 기반의 시드를 사용해 평가용 정수 시드 리스트를 생성
    scope에는 문맥(예: "noise|metric")을 넣어 조합별로 다른 시드가 나오게 함
    ret_bits는 시드의 최대값 범위를 제어하며, 결과 시드는 [0, 2**ret_bits - 1] 범위
    """
    n = max(1, int(n))
    base = int(base_seed)
    # scope + base_seed로부터 결정적인 64비트 시드를 도출
    root = hash_to_int(f"{scope}|{base}", bits=64)
    rng = np.random.Generator(np.random.PCG64(root))
    high = (1 << ret_bits) - 1
    arr = rng.integers(0, high, size=n, dtype=np.int64)
    return arr.astype(int).tolist()
