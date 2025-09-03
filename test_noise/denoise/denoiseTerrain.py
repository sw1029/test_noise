from .base import Denoise
from ..utils import *
import numpy as np
import yaml
import os

class DenoiseTerrain(Denoise):
    @staticmethod
    def denoise(src, DEM=None, pixel_size=1.0,
                  sun_angle=30, sun_azimuth=225, sun_elevation=45, 
                  factor=0.1, slope=30, max_slope=45,
                  Minnaert_constant_NIR=0.6,
                  Minnaert_constant_R=0.5,
                  Minnaert_constant_G=0.4,
                  Minnaert_constant_B=0.3,
                  yaml_name="KOMPSAT.yaml") -> np.ndarray:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, '..', 'config', yaml_name)
        _sun_angle = sun_angle
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        band_params = config.get('band', {})
        
        gain_B = band_params.get('blue', {}).get('gain')
        offset_B = band_params.get('blue', {}).get('offset')
        gain_G = band_params.get('green', {}).get('gain')
        offset_G = band_params.get('green', {}).get('offset')
        gain_R = band_params.get('red', {}).get('gain')
        offset_R = band_params.get('red', {}).get('offset')
        gain_NIR = band_params.get('nir', {}).get('gain')
        offset_NIR = band_params.get('nir', {}).get('offset')

        rows, cols, channels = src.shape
        orig_dtype = src.dtype
        img = src.astype(np.float32, copy=False)
        terrain_denoise_image = img.copy()

        if DEM is not None: # DEM 이용 단순 보정
            _sun_angle, _slope = angle(DEM, sun_azimuth, sun_elevation, pixel_size=pixel_size)
        else:
            _slope = slope
        
        # 해당 radiance 값에 음수 clipping 적용 시 단색 이미지가 반환되는 오류가 발생.
        radiance_B = DN2radiance(img[:,:,0], gain_B, offset_B)
        radiance_G = DN2radiance(img[:,:,1], gain_G, offset_G)
        radiance_R = DN2radiance(img[:,:,2], gain_R, offset_R)

        terrain_denoise_image[:, :, 0] = Minnaert(radiance_B, _sun_angle, _slope, Minnaert_constant_B)
        terrain_denoise_image[:, :, 1] = Minnaert(radiance_G, _sun_angle, _slope, Minnaert_constant_G)
        terrain_denoise_image[:, :, 2] = Minnaert(radiance_R, _sun_angle, _slope, Minnaert_constant_R)

        terrain_denoise_image[:, :, 0] = radiance2DN(terrain_denoise_image[:, :, 0], gain_B, offset_B)
        terrain_denoise_image[:, :, 1] = radiance2DN(terrain_denoise_image[:, :, 1], gain_G, offset_G)
        terrain_denoise_image[:, :, 2] = radiance2DN(terrain_denoise_image[:, :, 2], gain_R, offset_R)
        
        if channels == 4:
            radiance_NIR = DN2radiance(img[:, :, 3], gain_NIR, offset_NIR)
            terrain_denoise_image[:, :, 3] = Minnaert(radiance_NIR, _sun_angle, _slope, Minnaert_constant_NIR)
            terrain_denoise_image[:, :, 3] = radiance2DN(terrain_denoise_image[:, :, 3], gain_NIR, offset_NIR)
       
        # 안전한 범위/타입으로 반환
        terrain_denoise_image = np.nan_to_num(terrain_denoise_image)
        if np.issubdtype(orig_dtype, np.integer):
            info = np.iinfo(orig_dtype)
            terrain_denoise_image = np.clip(terrain_denoise_image, info.min, info.max).astype(orig_dtype)
        else:
            terrain_denoise_image = terrain_denoise_image.astype(np.float32)
        return terrain_denoise_image
    @staticmethod
    def denoise_topo(src: np.ndarray,
                     DEM: np.ndarray | None = None,
                     sun_azimuth: float = 225.0,
                     sun_elevation: float = 45.0,
                     pixel_size: float | tuple[float, float] = 1.0,
                     eps: float = 1e-3,
                     sample_max: int = 200_000,
                     mode: str = "luminance",
                     method: str = "c",
                     scale_clip: tuple[float, float] | None = (0.5, 1.5),
                     shadow_thresh: float | None = 0.2,
                     robust: bool = True,
                     ransac_iter: int = 200,
                     inlier_sigma: float = 2.5,
                     mask: np.ndarray | None = None) -> np.ndarray:
        """
        C-correction 기반 지형 보정(게인/오프셋 불요): DenoiseTerrain 보조 함수
        - 휘도 기반 단일 스케일 또는 채널별(per_channel) 방식 선택
        - RANSAC 기반 강건 회귀, 그림자 보호, 보정 강도 클리핑 제공

        참고 레퍼런스
        - https://github.com/OSGeo/grass/blob/main/imagery/i.topo.corr/i.topo.corr.md
        - https://grass.osgeo.org/grass-stable/manuals/i.topo.corr.html
        """
        if DEM is None:
            return src

        orig_dtype = src.dtype
        img = src.astype(np.float32, copy=False)

        i_deg, _ = angle(DEM, sun_azimuth, sun_elevation, pixel_size=pixel_size)
        cosI = np.cos(np.deg2rad(i_deg))
        cosZ = np.sin(np.deg2rad(sun_elevation))

        valid = np.isfinite(cosI) & (cosI > eps)
        if mask is not None:
            try:
                valid &= mask.astype(bool)
            except Exception:
                pass
        if not np.any(valid):
            return src

        H, W, C = img.shape
        out = img.copy()

        idx = np.flatnonzero(valid)
        if idx.size > sample_max:
            sel = np.random.default_rng(0).choice(idx, size=sample_max, replace=False)
        else:
            sel = idx

        cosI_s = cosI.ravel()[sel].reshape(-1, 1)
        X = np.concatenate([cosI_s, np.ones_like(cosI_s)], axis=1)  # [cosI, 1]
        XtX = X.T @ X

        def fit_ab(Xm: np.ndarray, ym: np.ndarray) -> tuple[float, float]:
            # y = a*cosI + b 선형 적합
            if not robust:
                try:
                    beta = np.linalg.solve(XtX, X.T @ ym)
                    return float(beta[0]), float(beta[1])
                except np.linalg.LinAlgError:
                    return 1.0, 0.0
            # RANSAC
            try:
                beta_ols = np.linalg.solve(XtX, X.T @ ym)
            except np.linalg.LinAlgError:
                beta_ols = np.array([[1.0], [0.0]], dtype=np.float32)
            a_best = float(beta_ols[0]); b_best = float(beta_ols[1])
            inliers_best = np.ones(ym.shape[0], dtype=bool)
            rng = np.random.default_rng(0)
            n = ym.shape[0]
            for _ in range(ransac_iter):
                if n < 2:
                    break
                idx2 = rng.choice(n, size=2, replace=False)
                X2 = X[idx2]
                y2 = ym[idx2]
                try:
                    beta2 = np.linalg.solve(X2, y2)
                    a2 = float(beta2[0]); b2 = float(beta2[1])
                except np.linalg.LinAlgError:
                    continue
                resid = (X @ beta2 - ym).ravel()
                std = float(np.std(resid)) + 1e-6
                thr = inlier_sigma * std
                inliers = np.abs(resid) <= thr
                if inliers.sum() > inliers_best.sum():
                    inliers_best = inliers
                    a_best, b_best = a2, b2
            try:
                Xin = X[inliers_best]
                yin = ym[inliers_best]
                beta = np.linalg.solve(Xin.T @ Xin, Xin.T @ yin)
                return float(beta[0]), float(beta[1])
            except np.linalg.LinAlgError:
                return a_best, b_best

        method_l = (method or "").lower()
        if method_l == "minnaert":
            # Minnaert 보정: Lcorr = L * (cosZ / cosI)^k, k는 회귀로 추정 (휘도 기반)
            if C >= 3:
                Y = 0.114 * img[..., 0] + 0.587 * img[..., 1] + 0.299 * img[..., 2]
            else:
                Y = img[..., 0]

            # 유효 샘플: cosI>eps, Y>0
            valid_k = (cosI.ravel()[sel] > eps)
            y_samp = Y.ravel()[sel][valid_k]
            x_samp = cosI.ravel()[sel][valid_k]
            # 로그-선형화: log(Y) = k * log(cosI) + const (대략적 근사)
            y_samp = np.clip(y_samp, eps, None)
            x_samp = np.clip(x_samp, eps, None)
            Xk = np.vstack([np.log(x_samp), np.ones_like(x_samp)]).T
            betak, _, _, _ = np.linalg.lstsq(Xk, np.log(y_samp), rcond=None)
            k_est = float(betak[0]) if np.isfinite(betak[0]) else 0.5

            # 스케일 계산 및 적용
            denom = np.maximum(cosI, eps)
            scale = (np.sin(np.deg2rad(sun_elevation)) / denom) ** k_est
            if shadow_thresh is not None:
                mask_shadow = cosI < shadow_thresh
                scale = np.where(mask_shadow, 1.0 + 0.5 * (scale - 1.0), scale)
            if scale_clip is not None:
                smin, smax = scale_clip
                scale = np.clip(scale, smin, smax)
            for ch in range(C):
                out[..., ch] = img[..., ch] * scale

        elif method_l in ("c-per_channel", "per_channel") or (method_l not in ("c", "c-luminance") and mode == "per_channel"):
            for ch in range(C):
                L = img[..., ch]
                y = L.ravel()[sel].reshape(-1, 1)
                a, b = fit_ab(X, y)

                c = max((b / a) if abs(a) > eps else 0.0, 0.0)
                denom = cosI + c
                scale = (cosZ + c) / np.maximum(denom, eps)
                if shadow_thresh is not None:
                    mask_shadow = cosI < shadow_thresh
                    scale = np.where(mask_shadow, 1.0 + 0.5 * (scale - 1.0), scale)
                if scale_clip is not None:
                    smin, smax = scale_clip
                    scale = np.clip(scale, smin, smax)
                out[..., ch] = L * scale
        else:  # C-correction, luminance 기반
            # 휘도 기반(색 보존)
            if C >= 3:
                Y = 0.114 * img[..., 0] + 0.587 * img[..., 1] + 0.299 * img[..., 2]
            else:
                Y = img[..., 0]
            y = Y.ravel()[sel].reshape(-1, 1)
            a, b = fit_ab(X, y)

            c = max((b / a) if abs(a) > eps else 0.0, 0.0)
            denom = cosI + c
            scale = (cosZ + c) / np.maximum(denom, eps)
            if shadow_thresh is not None:
                mask_shadow = cosI < shadow_thresh
                scale = np.where(mask_shadow, 1.0 + 0.5 * (scale - 1.0), scale)
            if scale_clip is not None:
                smin, smax = scale_clip
                scale = np.clip(scale, smin, smax)
            for ch in range(C):
                out[..., ch] = img[..., ch] * scale

        # 출력 dtype 처리
        out = np.nan_to_num(out)
        if np.issubdtype(orig_dtype, np.integer):
            info = np.iinfo(orig_dtype)
            out = np.clip(out, info.min, info.max).astype(orig_dtype)
        return out
