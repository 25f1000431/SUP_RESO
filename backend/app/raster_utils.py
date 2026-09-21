"""
GeoTIFF reading/writing: band validation, normalization, per-band
statistics (used for the diagnostics panel), and saving the
super-resolved output.
"""

from pathlib import Path

from fastapi import HTTPException
import numpy as np
import rasterio
from rasterio.transform import Affine

from app.config import BAND_NAMES

REQUIRED_BANDS = {"B02", "B03", "B04", "B08"}


def band_stats(band: np.ndarray) -> dict:
    finite = band[np.isfinite(band)]
    if finite.size == 0:
        return {}

    return {
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "p01": float(np.percentile(finite, 1)),
        "p50": float(np.percentile(finite, 50)),
        "p99": float(np.percentile(finite, 99)),
        "zero_percent": float(np.mean(finite == 0) * 100),
    }


def all_band_stats(arr: np.ndarray) -> dict:
    return {name: band_stats(arr[i]) for i, name in enumerate(BAND_NAMES)}


def validate_raster(src: rasterio.DatasetReader) -> dict:
    if src.count != 4:
        raise HTTPException(status_code=400, detail=f"Expected 4 bands, found {src.count}.")

    if src.crs is None:
        raise HTTPException(status_code=400, detail="GeoTIFF has no CRS information.")

    xres, yres = abs(src.transform.a), abs(src.transform.e)
    if not (8.0 <= xres <= 12.0 and 8.0 <= yres <= 12.0):
        raise HTTPException(
            status_code=400,
            detail=f"Expected ~10m resolution, detected {xres:.2f}m x {yres:.2f}m.",
        )

    descriptions = [str(d).strip().upper() if d else "" for d in (src.descriptions or [])]

    if REQUIRED_BANDS.issubset(set(descriptions)):
        return {name: i + 1 for i, name in enumerate(descriptions)}

    if all(d == "" for d in descriptions):
        # No band metadata -- assume the standard B04/B03/B02/B08 order.
        return {"B04": 1, "B03": 2, "B02": 3, "B08": 4}

    raise HTTPException(
        status_code=400,
        detail="GeoTIFF has 4 bands but metadata doesn't identify B02/B03/B04/B08.",
    )


def read_rgbn(path: Path):
    """Read a Sentinel-2 GeoTIFF and return a normalized (4, H, W) array."""

    with rasterio.open(path) as src:
        band_map = validate_raster(src)
        band_order = [band_map[name] for name in BAND_NAMES]

        arr = src.read(band_order).astype(np.float32)
        profile, transform, crs = src.profile.copy(), src.transform, src.crs

    raw_stats = all_band_stats(arr)

    finite_values = arr[np.isfinite(arr)]
    if finite_values.size == 0:
        raise HTTPException(status_code=400, detail="Image contains no finite values.")

    # Sentinel-2 L2A digital numbers are scaled by 10000; reflectance is 0-1.
    normalization_applied = float(np.percentile(finite_values, 99.9)) > 1.5
    if normalization_applied:
        arr = arr / 10000.0

    arr = np.clip(np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    normalized_stats = all_band_stats(arr)

    return arr, profile, transform, crs, raw_stats, normalized_stats, normalization_applied


def save_sr_geotiff(sr: np.ndarray, profile: dict, transform, crs, output_path: Path) -> None:
    output_transform = transform * Affine.scale(0.25, 0.25)

    profile.update(
        driver="GTiff",
        height=sr.shape[1],
        width=sr.shape[2],
        count=4,
        dtype="float32",
        transform=output_transform,
        crs=crs,
        compress="deflate",
    )
    for key in ("blockxsize", "blockysize", "tiled"):
        profile.pop(key, None)

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(sr.astype(np.float32))
        for i, name in enumerate(BAND_NAMES, start=1):
            dst.set_band_description(i, name)
