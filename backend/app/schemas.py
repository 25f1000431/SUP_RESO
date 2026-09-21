"""Request/response schemas."""

from pydantic import BaseModel


class AOIRequest(BaseModel):
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float

    start_date: str | None = None
    end_date: str | None = None
    max_cloud_coverage: float = 20.0
