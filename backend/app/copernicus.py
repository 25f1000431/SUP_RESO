"""
Everything related to talking to Copernicus Data Space: OAuth token
caching, AOI geometry, and downloading a Sentinel-2 GeoTIFF.
"""

from datetime import date, timedelta
from pathlib import Path
import math
import time

from fastapi import HTTPException
import requests
from rasterio.warp import transform_bounds

from app.config import (
    COPERNICUS_CLIENT_ID,
    COPERNICUS_CLIENT_SECRET,
    COPERNICUS_TOKEN_URL,
    COPERNICUS_PROCESS_URL,
    MAX_AOI_PIXELS,
)
from app.schemas import AOIRequest

EVALSCRIPT = """
//VERSION=3
function setup() {
    return {
        input: [{ bands: ["B02", "B03", "B04", "B08"], units: "REFLECTANCE" }],
        output: { bands: 4, sampleType: SampleType.FLOAT32 }
    };
}
function evaluatePixel(sample) {
    return [sample.B04, sample.B03, sample.B02, sample.B08];
}
"""

_token = None
_token_expires_at = 0


def get_token() -> str:
    """Return a cached OAuth token, refreshing it if expired."""
    global _token, _token_expires_at

    if _token and time.time() < _token_expires_at:
        return _token

    if not COPERNICUS_CLIENT_ID or not COPERNICUS_CLIENT_SECRET:
        raise HTTPException(
            status_code=500,
            detail="COPERNICUS_CLIENT_ID / COPERNICUS_CLIENT_SECRET not configured.",
        )

    try:
        response = requests.post(
            COPERNICUS_TOKEN_URL,
            data={
                "grant_type": "client_credentials",
                "client_id": COPERNICUS_CLIENT_ID,
                "client_secret": COPERNICUS_CLIENT_SECRET,
            },
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        _token = data["access_token"]
        _token_expires_at = time.time() + max(int(data.get("expires_in", 300)) - 60, 30)
        return _token

    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Copernicus auth failed: {exc}")


def get_date_range(request: AOIRequest) -> tuple[date, date]:
    try:
        end_date = date.fromisoformat(request.end_date) if request.end_date else date.today()
        start_date = (
            date.fromisoformat(request.start_date)
            if request.start_date
            else end_date - timedelta(days=30)
        )
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date. Use YYYY-MM-DD.")

    if start_date > end_date:
        raise HTTPException(status_code=400, detail="start_date cannot be after end_date.")

    return start_date, end_date


def get_utm_epsg(longitude: float, latitude: float) -> int:
    zone = max(1, min(int((longitude + 180) // 6) + 1, 60))
    return (32600 if latitude >= 0 else 32700) + zone


def calculate_aoi_dimensions(min_lon, min_lat, max_lon, max_lat):
    center_lon, center_lat = (min_lon + max_lon) / 2, (min_lat + max_lat) / 2
    epsg = get_utm_epsg(center_lon, center_lat)

    left, bottom, right, top = transform_bounds(
        "EPSG:4326", f"EPSG:{epsg}", min_lon, min_lat, max_lon, max_lat, densify_pts=21
    )

    width = max(1, math.ceil((right - left) / 10.0))
    height = max(1, math.ceil((top - bottom) / 10.0))

    return epsg, (left, bottom, right, top), width, height


def download_sentinel2_aoi(request: AOIRequest, output_path: Path) -> dict:
    token = get_token()
    start_date, end_date = get_date_range(request)

    epsg, (left, bottom, right, top), width, height = calculate_aoi_dimensions(
        request.min_lon, request.min_lat, request.max_lon, request.max_lat
    )

    print(f"[COPERNICUS] {start_date} → {end_date}, cloud<{request.max_cloud_coverage}%, "
          f"{width}x{height}px, EPSG:{epsg}")

    if width > MAX_AOI_PIXELS or height > MAX_AOI_PIXELS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Selected AOI is too large ({width}x{height}px at 10m). "
                "Please select an area smaller than ~10km x 10km."
            ),
        )

    process_request = {
        "input": {
            "bounds": {
                "properties": {"crs": f"http://www.opengis.net/def/crs/EPSG/0/{epsg}"},
                "bbox": [left, bottom, right, top],
            },
            "data": [{
                "type": "sentinel-2-l2a",
                "dataFilter": {
                    "timeRange": {
                        "from": f"{start_date}T00:00:00Z",
                        "to": f"{end_date}T23:59:59Z",
                    },
                    "maxCloudCoverage": request.max_cloud_coverage,
                    "mosaickingOrder": "leastCC",
                },
            }],
        },
        "output": {
            "width": width,
            "height": height,
            "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}],
        },
        "evalscript": EVALSCRIPT,
    }

    try:
        response = requests.post(
            COPERNICUS_PROCESS_URL,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Accept": "image/tiff",
            },
            json=process_request,
            timeout=180,
        )

        if not response.ok:
            raise HTTPException(
                status_code=502,
                detail=f"Copernicus returned {response.status_code}: {response.text[:2000]}",
            )

        if not response.content:
            raise HTTPException(status_code=502, detail="Copernicus returned an empty image.")

        output_path.write_bytes(response.content)
        print(f"[COPERNICUS] Downloaded {len(response.content):,} bytes")

        return {
            "epsg": epsg,
            "width": width,
            "height": height,
            "start_date": str(start_date),
            "end_date": str(end_date),
            "max_cloud_coverage": request.max_cloud_coverage,
        }

    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Could not reach Copernicus: {exc}")
