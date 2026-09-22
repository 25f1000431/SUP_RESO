"""
The two super-resolution endpoints:
  POST /api/super-resolve-area    -- download a Sentinel-2 AOI, then SR it
  POST /api/super-resolve         -- SR an uploaded GeoTIFF directly
"""

from pathlib import Path
import tempfile
import uuid

from fastapi import APIRouter, HTTPException, UploadFile, File
import numpy as np

from app.config import DEVICE, OUTPUT_DIR, BAND_NAMES
from app.schemas import AOIRequest
from app.validators import validate_aoi
from app import model_loader
from app.copernicus import download_sentinel2_aoi
from app.raster_utils import read_rgbn, save_sr_geotiff
from app.preview import make_rgb_preview, as_data_uri
from app.inference import run_super_resolution

from app.segmentation import DeepLabSegmenter

router = APIRouter(prefix="/api")
segmenter = DeepLabSegmenter(device=DEVICE)

def _build_diagnostics(arr, sr, raw_stats, normalized_stats, output_stats,
                        normalization_applied, clipping) -> dict:
    return {
        "normalization": {"divided_by_10000": normalization_applied},
        "input_global": {
            "min": float(np.min(arr)), "max": float(np.max(arr)), "mean": float(np.mean(arr)),
        },
        "output_global": {
            "min": float(np.min(sr)), "max": float(np.max(sr)), "mean": float(np.mean(sr)),
        },
        "raw_bands": raw_stats,
        "normalized_bands": normalized_stats,
        "output_bands": output_stats,
        "clipping": clipping,
    }


@router.post("/super-resolve-area")
async def super_resolve_area(request: AOIRequest):
    model_loader.require_model()
    validate_aoi(request)

    job_id = uuid.uuid4().hex
    input_path = OUTPUT_DIR / f"{job_id}_sentinel2_input.tif"

    output_filename = f"{job_id}_sr.tif"
    output_path = OUTPUT_DIR / output_filename

    segmentation_filename = f"{job_id}_segmentation.tif"
    segmentation_path = OUTPUT_DIR / segmentation_filename

    try:
        # -----------------------------------------------------
        # 1. Download Sentinel-2
        # -----------------------------------------------------

        download_info = download_sentinel2_aoi(
            request,
            input_path,
        )

        # -----------------------------------------------------
        # 2. Read 4-band Sentinel-2
        # -----------------------------------------------------

        (
            arr,
            profile,
            transform,
            crs,
            raw_stats,
            normalized_stats,
            normalization_applied,
        ) = read_rgbn(input_path)

        # -----------------------------------------------------
        # 3. Original preview
        # -----------------------------------------------------

        original_preview = make_rgb_preview(
            arr,
            is_original=True,
        )

        # -----------------------------------------------------
        # 4. SEN2SR
        # -----------------------------------------------------

        sr, output_stats, clipping = run_super_resolution(arr)

        # -----------------------------------------------------
        # 5. Save real 4-band SR GeoTIFF
        # -----------------------------------------------------

        save_sr_geotiff(
            sr,
            profile,
            transform,
            crs,
            output_path,
        )

        # -----------------------------------------------------
        # 6. SR preview
        # -----------------------------------------------------

        sr_preview = make_rgb_preview(sr)

        # -----------------------------------------------------
        # 7. DeepLabV3+ segmentation
        # -----------------------------------------------------

        segmentation_result = segmenter.predict(
            str(input_path),
            str(segmentation_path),
        )

        # -----------------------------------------------------
        # 8. Response
        # -----------------------------------------------------

        return {
            "success": True,

            "source": "Copernicus Data Space",

            "device": str(DEVICE),

            "aoi": {
                "min_lon": request.min_lon,
                "min_lat": request.min_lat,
                "max_lon": request.max_lon,
                "max_lat": request.max_lat,
            },

            "sentinel2": {
                "collection": "sentinel-2-l2a",
                "bands": [
                    "B02",
                    "B03",
                    "B04",
                    "B08",
                ],
                "model_band_order": BAND_NAMES,
                "resolution_m": 10,
                "width": int(arr.shape[2]),
                "height": int(arr.shape[1]),
                "start_date": download_info["start_date"],
                "end_date": download_info["end_date"],
                "max_cloud_coverage": download_info[
                    "max_cloud_coverage"
                ],
                "crs": str(crs),
            },

            "output": {
                "width": int(sr.shape[2]),
                "height": int(sr.shape[1]),
                "resolution_m": 2.5,
                "bands": BAND_NAMES,
                "download_url": f"/outputs/{output_filename}",
            },

            "original_preview": as_data_uri(
                original_preview
            ),

            "sr_preview": as_data_uri(
                sr_preview
            ),

            "segmentation": {
                "status": "completed",

                "download_url": f"/outputs/{segmentation_filename}",

                "preview_url": (
                    f"/outputs/"
                    f"{Path(segmentation_result['segmentation_preview_path']).name}"
                ),

                "ndvi_preview_url": (
                    f"/outputs/"
                    f"{Path(segmentation_result['ndvi_preview_path']).name}"
                ),
                "health_preview_url": (
                    f"/outputs/"
                    f"{Path(segmentation_result['health_preview_path']).name}"
                ),

                **segmentation_result,
            },


            "diagnostics": _build_diagnostics(
                arr,
                sr,
                raw_stats,
                normalized_stats,
                output_stats,
                normalization_applied,
                clipping,
            ),
        }

    except HTTPException:
        raise

    except Exception as exc:
        print(
            f"[ERROR] super-resolve-area failed: "
            f"{exc!r}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                f"Super-resolution/segmentation failed: "
                f"{exc}"
            ),
        )

    finally:
        input_path.unlink(
            missing_ok=True
        )





@router.post("/super-resolve")
async def super_resolve(file: UploadFile = File(...)):
    model_loader.require_model()

    if not file.filename or not file.filename.lower().endswith((".tif", ".tiff")):
        raise HTTPException(status_code=400, detail="Please upload a GeoTIFF (.tif/.tiff).")

    job_id = uuid.uuid4().hex

    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp.write(await file.read())
        input_path = Path(tmp.name)

    try:
        arr, profile, transform, crs, raw_stats, normalized_stats, normalization_applied = (
            read_rgbn(input_path)
        )

        if arr.shape[1] > 1024 or arr.shape[2] > 1024:
            raise HTTPException(status_code=400, detail="Input limit is 1024x1024 pixels.")

        original_preview = make_rgb_preview(arr)
        sr, output_stats, clipping = run_super_resolution(arr)

        output_filename = f"{job_id}_sr.tif"
        output_path = OUTPUT_DIR / output_filename
        save_sr_geotiff(sr, profile, transform, crs, output_path)

        sr_preview = make_rgb_preview(sr)

        return {
            "success": True,
            "device": str(DEVICE),
            "input": {
                "filename": file.filename,
                "width": int(arr.shape[2]),
                "height": int(arr.shape[1]),
                "resolution_m": 10,
                "bands": BAND_NAMES,
            },
            "output": {
                "width": int(sr.shape[2]),
                "height": int(sr.shape[1]),
                "resolution_m": 2.5,
                "bands": BAND_NAMES,
                "download_url": f"/outputs/{output_filename}",
            },
            "original_preview": as_data_uri(original_preview),
            "sr_preview": as_data_uri(sr_preview),
            "diagnostics": _build_diagnostics(
                arr, sr, raw_stats, normalized_stats, output_stats,
                normalization_applied, clipping,
            ),
        }

    finally:
        input_path.unlink(missing_ok=True)
