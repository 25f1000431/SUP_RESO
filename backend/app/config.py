"""
Central configuration: paths, environment variables, device.
Everything else in the app imports settings from here.
"""

from pathlib import Path
import os

import torch
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent.parent

load_dotenv(BASE_DIR / "backend" / ".env")

# --- Copernicus Data Space -------------------------------------------------

COPERNICUS_CLIENT_ID = os.getenv("COPERNICUS_CLIENT_ID")
COPERNICUS_CLIENT_SECRET = os.getenv("COPERNICUS_CLIENT_SECRET")

COPERNICUS_TOKEN_URL = os.getenv(
    "COPERNICUS_TOKEN_URL",
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
)

COPERNICUS_PROCESS_URL = os.getenv(
    "COPERNICUS_PROCESS_URL",
    "https://sh.dataspace.copernicus.eu/process/v1",
)

# --- Paths -------------------------------------------------------------

MODEL_PATH = BASE_DIR / "models" / "sen2sr_full_2851roi.pth"
OUTPUT_DIR = BASE_DIR / "backend" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Device --------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Limits ----------------------------------------------------------------

MAX_AOI_PIXELS = 1024  # max width/height at 10m resolution
BAND_NAMES = ["B04", "B03", "B02", "B08"]  # model band order
