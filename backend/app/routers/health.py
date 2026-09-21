from fastapi import APIRouter

from app import model_loader
from app.config import DEVICE, COPERNICUS_CLIENT_ID, COPERNICUS_CLIENT_SECRET

router = APIRouter()


@router.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "model_loaded": model_loader.model is not None,
        "model_load_error": model_loader.load_error,
        "copernicus_configured": bool(COPERNICUS_CLIENT_ID and COPERNICUS_CLIENT_SECRET),
    }
