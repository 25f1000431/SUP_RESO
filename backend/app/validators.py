"""Input validation for area-of-interest requests."""

from fastapi import HTTPException

from app.schemas import AOIRequest


def validate_aoi(request: AOIRequest) -> None:
    if not (-180 <= request.min_lon <= 180 and -180 <= request.max_lon <= 180):
        raise HTTPException(status_code=400, detail="Longitude must be between -180 and 180.")

    if not (-90 <= request.min_lat <= 90 and -90 <= request.max_lat <= 90):
        raise HTTPException(status_code=400, detail="Latitude must be between -90 and 90.")

    if request.min_lon >= request.max_lon:
        raise HTTPException(status_code=400, detail="min_lon must be smaller than max_lon.")

    if request.min_lat >= request.max_lat:
        raise HTTPException(status_code=400, detail="min_lat must be smaller than max_lat.")

    if not (0 <= request.max_cloud_coverage <= 100):
        raise HTTPException(status_code=400, detail="max_cloud_coverage must be 0-100.")
