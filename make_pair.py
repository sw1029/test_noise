import os
from pathlib import Path
import io
import math
import gzip
import requests
from typing import List, Tuple

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.merge import merge as rio_merge
from rasterio.transform import Affine
from rasterio.coords import BoundingBox

import planetary_computer as pc
from pystac_client import Client
import xarray as xr
import stackstac


# ---------------------------
# 설정값
# ---------------------------
OUT_DIR = Path("data"); OUT_DIR.mkdir(exist_ok=True)
CENTER_LAT, CENTER_LON = 37.5, 127.0
HALF_SIZE_DEG = 0.05
DATE = "2023-06-01/2023-08-31"
MAX_CLOUD = 20  # %
TARGET_EPSG = 4326  # 간단히 WGS84; (서울 UTM은 32652)
RGB_BANDS = ["red", "green", "blue"]

# SRTM(HGT) 공개 버킷(1 arc-second, ~30 m)
# 타일 경로 규칙: https://s3.amazonaws.com/elevation-tiles-prod/skadi/N37/N37E127.hgt.gz
SRTM_SKADI = "https://s3.amazonaws.com/elevation-tiles-prod/skadi/{ns}{ilat:02d}/{ns}{ilat:02d}{ew}{ilon:03d}.hgt.gz"


def bbox_deg(center_lat: float, center_lon: float, half_size_deg: float) -> Tuple[float, float, float, float]:
    minx = center_lon - half_size_deg
    maxx = center_lon + half_size_deg
    miny = center_lat - half_size_deg
    maxy = center_lat + half_size_deg
    return minx, miny, maxx, maxy


def _tile_indices_from_bbox(minx: float, miny: float, maxx: float, maxy: float) -> List[Tuple[int, int]]:
    """HGT 1x1도 타일 인덱스(정수 위경도) 목록 생성"""
    lons = range(math.floor(minx), math.ceil(maxx))
    lats = range(math.floor(miny), math.ceil(maxy))
    return [(lat, lon) for lat in lats for lon in lons]


def _tile_url(lat: int, lon: int) -> str:
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return SRTM_SKADI.format(ns=ns, ilat=abs(lat), ew=ew, ilon=abs(lon))


def _download_and_open_hgtgz(lat: int, lon: int, tmp_dir: Path):
    """HGT.GZ를 내려받아 메모리에서 해제 후 rasterio로 오픈 가능한 임시 TIFF로 변환 없이 바로 열기 시도.
       GDAL은 HGT를 직접 열 수 있으므로 rasterio.open(hgt_bytes) 사용.
       일부 환경에서 직접 open이 어려우면, 임시 파일로 저장 후 open."""
    url = _tile_url(lat, lon)
    resp = requests.get(url, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"Failed to fetch {url} (status {resp.status_code})")
    # .hgt.gz -> .hgt 바이트
    hgt_bytes = gzip.decompress(resp.content)

    # 임시 파일로 저장 후 오픈 (대부분 환경에서 안정적)
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    hgt_path = tmp_dir / f"{ns}{abs(lat):02d}{ew}{abs(lon):03d}.hgt"
    with open(hgt_path, "wb") as f:
        f.write(hgt_bytes)

    # rasterio가 HGT를 GDAL 통해 인식
    ds = rasterio.open(hgt_path)  # 드라이버=SRTMHGT
    return ds


def build_srtm_dem_from_bbox(minx: float, miny: float, maxx: float, maxy: float, out_path: Path) -> None:
    """GDAL 미설치 환경에서도 동작: 필요한 HGT 타일들을 직접 받아 모자이크 후 bbox로 클립하여 GeoTIFF 저장."""
    tmp_dir = OUT_DIR / "_tmp_hgt"
    tmp_dir.mkdir(exist_ok=True)

    # 1) 커버하는 HGT 타일 수집
    tiles = _tile_indices_from_bbox(minx, miny, maxx, maxy)
    if not tiles:
        raise RuntimeError("No tiles computed for bbox.")

    datasets = []
    try:
        for lat, lon in tiles:
            ds = _download_and_open_hgtgz(lat, lon, tmp_dir)
            datasets.append(ds)

        # 2) 타일 모자이크 (자연 좌표계 WGS84)
        mosaic, mosaic_transform = rio_merge(datasets)
        # mosaic shape: (1, H, W)
        crs = datasets[0].crs

        # 3) bbox로 클립 (경위도 bbox -> window)
        # rasterio.windows.from_bounds expects bounds in CRS of dataset (WGS84)
        window = rasterio.windows.from_bounds(
            left=minx, bottom=miny, right=maxx, top=maxy,
            transform=mosaic_transform, height=mosaic.shape[1], width=mosaic.shape[2]
        )
        # window로 슬라이싱
        row_off, col_off = int(window.row_off), int(window.col_off)
        height, width = int(window.height), int(window.width)
        clipped = mosaic[:, row_off:row_off+height, col_off:col_off+width]
        clipped_transform = mosaic_transform * Affine.translation(col_off, row_off)

        profile = {
            "driver": "GTiff",
            "height": clipped.shape[1],
            "width": clipped.shape[2],
            "count": 1,
            "dtype": "int16",  # HGT 기본은 int16 (해수면 미측정은 -32768)
            "crs": crs,
            "transform": clipped_transform,
            "tiled": True,
            "compress": "deflate",
        }
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(clipped.astype("int16"))
    finally:
        # 열었던 데이터셋 닫기
        for ds in datasets:
            ds.close()
        # 임시 파일 정리는 상황에 따라 남겨두셔도 됩니다.


def main():
    # ---------------------------
    # 1) 관심영역(bbox)
    # ---------------------------
    minx, miny, maxx, maxy = bbox_deg(CENTER_LAT, CENTER_LON, HALF_SIZE_DEG)
    bbox = [minx, miny, maxx, maxy]

    print("Searching Landsat-8 scenes from Planetary Computer STAC...")
    client = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    search = client.search(
        collections=["landsat-c2-l2"],
        bbox=bbox,
        datetime=DATE,
        query={"eo:cloud_cover": {"lt": MAX_CLOUD}},
        limit=1,
    )
    items = list(search.items())
    if not items:
        raise SystemExit("조건에 맞는 Landsat-8 L2 장면이 없습니다. 날짜/구름 조건을 조정하세요.")

    item = items[0]
    print("Selected scene:", item.id)
    signed_item = pc.sign(item)

    # ---------------------------
    # 2) Landsat-8 RGB 스택 (stackstac) + CRS/transform 구성
    # ---------------------------
    print("Downloading and stacking Landsat-8 RGB bands...")
    stack = stackstac.stack(
        signed_item.to_dict(),
        assets=RGB_BANDS,
        bounds_latlon=bbox,
        epsg=TARGET_EPSG,   # CRS 지정 (STAC가 비어있을 수 있음)
        chunksize=2048,
    )

    arr = stack.compute().astype("float32")
    while "time" in arr.dims and arr.sizes.get("time", 1) == 1:
        arr = arr.isel(time=0)

    bands = [b for b in arr.band.values]
    assert all(b in RGB_BANDS for b in bands), f"예상 밴드 {RGB_BANDS}가 아닙니다: {bands}"
    rgb = np.stack([arr.sel(band=b).values for b in RGB_BANDS], axis=0)  # (3, y, x)

    # 보기 편한 간단 정규화
    def norm(a):
        lo, hi = np.nanpercentile(a, 2), np.nanpercentile(a, 98)
        a = np.clip(a, lo, hi)
        a = (a - a.min()) / (a.max() - a.min() + 1e-6)
        return a

    rgb = np.stack([norm(rgb[i]) for i in range(3)], axis=0).astype("float32")

    # 좌표축에서 transform 계산
    xs = arr.x.values
    ys = arr.y.values
    resx = float(xs[1] - xs[0])
    resy = float(ys[1] - ys[0])  # 보통 음수
    transform = Affine.translation(xs[0] - resx / 2.0, ys[0] - resy / 2.0) * Affine.scale(resx, resy)
    height, width = rgb.shape[1], rgb.shape[2]

    rgb_out = OUT_DIR / "sample_rgb.tif"
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 3,
        "dtype": "float32",
        "crs": rasterio.crs.CRS.from_epsg(TARGET_EPSG),
        "transform": transform,
        "tiled": True,
        "compress": "deflate",
    }
    with rasterio.open(rgb_out, "w", **profile) as dst:
        dst.write(rgb)
    print(f"RGB saved -> {rgb_out}")

    # ---------------------------
    # 3) DEM: GDAL 없는 환경용 순수 파이썬 SRTM 생성
    # ---------------------------
    print("Downloading/merging SRTM tiles (pure-Python path)...")
    dem_tmp = OUT_DIR / "srtm_bbox.tif"
    build_srtm_dem_from_bbox(minx, miny, maxx, maxy, dem_tmp)

    # ---------------------------
    # 4) DEM을 RGB 그리드/해상도로 재투영
    # ---------------------------
    dem_out = OUT_DIR / "sample_dem.tif"
    with rasterio.open(dem_tmp) as src_dem, rasterio.open(rgb_out) as ref:
        dst_profile = ref.profile.copy()
        dst_profile.update(count=1, dtype="float32", compress="deflate")
        with rasterio.open(dem_out, "w", **dst_profile) as dst:
            reproject(
                source=rasterio.band(src_dem, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src_dem.transform,
                src_crs=src_dem.crs,
                dst_transform=ref.transform,
                dst_crs=ref.crs,
                resampling=Resampling.bilinear,
            )
    print(f"DEM saved -> {dem_out}")

    print("\nDone. Generated pair:")
    print(" -", rgb_out.resolve())
    print(" -", dem_out.resolve())


if __name__ == "__main__":
    main()
