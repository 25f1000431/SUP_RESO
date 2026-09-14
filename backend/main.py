from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from pathlib import Path
from datetime import date, timedelta
import tempfile
import uuid
import base64
import io
import time
import math
import os

import numpy as np
import requests
import rasterio
import torch

from PIL import Image, ImageFilter, ImageEnhance
from rasterio.warp import transform_bounds
from dotenv import load_dotenv

from sen2sr.models.opensr_baseline.cnn import CNNSR


# ============================================================
# ENVIRONMENT
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(
    BASE_DIR / "backend" / ".env"
)

COPERNICUS_CLIENT_ID = os.getenv(
    "COPERNICUS_CLIENT_ID"
)

COPERNICUS_CLIENT_SECRET = os.getenv(
    "COPERNICUS_CLIENT_SECRET"
)

COPERNICUS_TOKEN_URL = os.getenv(
    "COPERNICUS_TOKEN_URL",
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
)

COPERNICUS_PROCESS_URL = os.getenv(
    "COPERNICUS_PROCESS_URL",
    "https://sh.dataspace.copernicus.eu/process/v1",
)


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "sen2sr_full_2851roi.pth"
)

OUTPUT_DIR = (
    BASE_DIR
    / "backend"
    / "outputs"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="SAT-SR",
    description="Sentinel-2 Super Resolution Prototype",
    version="2.1.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# OUTPUT FILES
# ============================================================

app.mount(
    "/outputs",
    StaticFiles(
        directory=str(OUTPUT_DIR)
    ),
    name="outputs",
)


# ============================================================
# MODEL
# ============================================================

model = None


def load_model():

    global model

    if not MODEL_PATH.exists():

        raise RuntimeError(
            f"Model checkpoint not found: "
            f"{MODEL_PATH}"
        )

    print()
    print("========================================")
    print("LOADING SAT-SR MODEL")
    print("========================================")
    print(
        f"Checkpoint: {MODEL_PATH}"
    )
    print(
        f"Device: {DEVICE}"
    )

    model = CNNSR(
        in_channels=4,
        out_channels=4,
        feature_channels=24,
        upscale=4,
        bias=True,
        train_mode=False,
        num_blocks=6,
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):

        state_dict = (
            checkpoint["model_state_dict"]
        )

    elif (
        isinstance(checkpoint, dict)
        and "state_dict" in checkpoint
    ):

        state_dict = (
            checkpoint["state_dict"]
        )

    else:

        state_dict = checkpoint

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.to(DEVICE)
    model.eval()

    print()
    print("========================================")
    print("SAT-SR MODEL LOADED")
    print("========================================")
    print(
        f"Device: {DEVICE}"
    )
    print(
        "Input channels: 4"
    )
    print(
        "Output channels: 4"
    )
    print(
        "Feature channels: 24"
    )
    print(
        "CNN blocks: 6"
    )
    print(
        "Upscale factor: 4x"
    )
    print("========================================")
    print()


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup_event():

    load_model()


# ============================================================
# COPERNICUS TOKEN CACHE
# ============================================================

copernicus_token = None
copernicus_token_expires_at = 0


def get_copernicus_token():

    global copernicus_token
    global copernicus_token_expires_at

    if (
        copernicus_token
        and
        time.time()
        <
        copernicus_token_expires_at
    ):

        return copernicus_token

    if not COPERNICUS_CLIENT_ID:

        raise HTTPException(
            status_code=500,
            detail=(
                "COPERNICUS_CLIENT_ID "
                "is not configured."
            ),
        )

    if not COPERNICUS_CLIENT_SECRET:

        raise HTTPException(
            status_code=500,
            detail=(
                "COPERNICUS_CLIENT_SECRET "
                "is not configured."
            ),
        )

    try:

        response = requests.post(
            COPERNICUS_TOKEN_URL,
            data={
                "grant_type":
                    "client_credentials",

                "client_id":
                    COPERNICUS_CLIENT_ID,

                "client_secret":
                    COPERNICUS_CLIENT_SECRET,
            },
            timeout=30,
        )

        response.raise_for_status()

        token_data = (
            response.json()
        )

        copernicus_token = (
            token_data[
                "access_token"
            ]
        )

        expires_in = int(
            token_data.get(
                "expires_in",
                300,
            )
        )

        copernicus_token_expires_at = (
            time.time()
            +
            max(
                expires_in - 60,
                30,
            )
        )

        return copernicus_token

    except requests.RequestException as exc:

        raise HTTPException(
            status_code=502,
            detail=(
                "Could not authenticate "
                "with Copernicus Data Space: "
                f"{exc}"
            ),
        )


# ============================================================
# AOI REQUEST
# ============================================================

class AOIRequest(BaseModel):

    min_lon: float
    min_lat: float

    max_lon: float
    max_lat: float

    start_date: str | None = None
    end_date: str | None = None

    max_cloud_coverage: float = 20.0


# ============================================================
# AOI VALIDATION
# ============================================================

def validate_aoi(
    request: AOIRequest
):

    if not (
        -180
        <= request.min_lon
        <= 180
        and
        -180
        <= request.max_lon
        <= 180
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Longitude must be "
                "between -180 and 180."
            ),
        )

    if not (
        -90
        <= request.min_lat
        <= 90
        and
        -90
        <= request.max_lat
        <= 90
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Latitude must be "
                "between -90 and 90."
            ),
        )

    if (
        request.min_lon
        >=
        request.max_lon
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "min_lon must be smaller "
                "than max_lon."
            ),
        )

    if (
        request.min_lat
        >=
        request.max_lat
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "min_lat must be smaller "
                "than max_lat."
            ),
        )

    if not (
        0
        <=
        request.max_cloud_coverage
        <=
        100
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "max_cloud_coverage must "
                "be between 0 and 100."
            ),
        )


# ============================================================
# DATE RANGE
# ============================================================

def get_date_range(
    request: AOIRequest
):

    if request.end_date:

        try:

            end_date = (
                date.fromisoformat(
                    request.end_date
                )
            )

        except ValueError:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Invalid end_date. "
                    "Use YYYY-MM-DD."
                ),
            )

    else:

        end_date = date.today()

    if request.start_date:

        try:

            start_date = (
                date.fromisoformat(
                    request.start_date
                )
            )

        except ValueError:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Invalid start_date. "
                    "Use YYYY-MM-DD."
                ),
            )

    else:

        start_date = (
            end_date
            -
            timedelta(days=30)
        )

    if start_date > end_date:

        raise HTTPException(
            status_code=400,
            detail=(
                "start_date cannot be "
                "after end_date."
            ),
        )

    return (
        start_date,
        end_date,
    )


# ============================================================
# UTM EPSG
# ============================================================

def get_utm_epsg(
    longitude,
    latitude,
):

    zone = (
        int(
            math.floor(
                (longitude + 180)
                / 6
            )
        )
        + 1
    )

    zone = max(
        1,
        min(zone, 60),
    )

    if latitude >= 0:

        return 32600 + zone

    return 32700 + zone


# ============================================================
# AOI DIMENSIONS
# ============================================================

def calculate_aoi_dimensions(
    min_lon,
    min_lat,
    max_lon,
    max_lat,
):

    center_lon = (
        min_lon
        +
        max_lon
    ) / 2

    center_lat = (
        min_lat
        +
        max_lat
    ) / 2

    epsg = get_utm_epsg(
        center_lon,
        center_lat,
    )

    utm_bounds = transform_bounds(
        "EPSG:4326",
        f"EPSG:{epsg}",
        min_lon,
        min_lat,
        max_lon,
        max_lat,
        densify_pts=21,
    )

    left, bottom, right, top = (
        utm_bounds
    )

    width = max(
        1,
        math.ceil(
            (right - left)
            / 10.0
        ),
    )

    height = max(
        1,
        math.ceil(
            (top - bottom)
            / 10.0
        ),
    )

    return (
        epsg,
        utm_bounds,
        width,
        height,
    )


# ============================================================
# COPERNICUS EVALSCRIPT
# ============================================================

COPERNICUS_EVALSCRIPT = """
//VERSION=3

function setup() {

    return {

        input: [

            {

                bands: [
                    "B02",
                    "B03",
                    "B04",
                    "B08"
                ],

                units: "REFLECTANCE"

            }

        ],

        output: {

            bands: 4,

            sampleType:
                SampleType.FLOAT32

        }

    };

}


function evaluatePixel(sample) {

    return [

        sample.B04,
        sample.B03,
        sample.B02,
        sample.B08

    ];

}
"""


# ============================================================
# DOWNLOAD SENTINEL-2
# ============================================================

def download_sentinel2_aoi(
    request: AOIRequest,
    output_path: Path,
):

    token = get_copernicus_token()

    start_date, end_date = (
        get_date_range(
            request
        )
    )

    (
        epsg,
        utm_bounds,
        width,
        height,
    ) = calculate_aoi_dimensions(
        request.min_lon,
        request.min_lat,
        request.max_lon,
        request.max_lat,
    )

    print()
    print("========================================")
    print("COPERNICUS REQUEST")
    print("========================================")
    print(
        f"Date range: {start_date} → {end_date}"
    )
    print(
        f"Cloud limit: "
        f"{request.max_cloud_coverage}%"
    )
    print(
        f"UTM EPSG: {epsg}"
    )
    print(
        f"Requested size: "
        f"{width} x {height}"
    )
    print(
        "Bands: B02 / B03 / B04 / B08"
    )
    print(
        "Model order: B04 / B03 / B02 / B08"
    )
    print("========================================")

    if (
        width > 1024
        or
        height > 1024
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Selected AOI is too large "
                "for the current prototype. "
                f"At 10m resolution it is "
                f"{width} x {height} pixels. "
                "Please select an area smaller "
                "than approximately 10 km x 10 km."
            ),
        )

    left, bottom, right, top = (
        utm_bounds
    )

    process_request = {

        "input": {

            "bounds": {

                "properties": {

                    "crs": (
                        "http://www.opengis.net/"
                        f"def/crs/EPSG/0/{epsg}"
                    )

                },

                "bbox": [

                    left,
                    bottom,
                    right,
                    top,

                ],

            },

            "data": [

                {

                    "type":
                        "sentinel-2-l2a",

                    "dataFilter": {

                        "timeRange": {

                            "from":
                                f"{start_date}"
                                "T00:00:00Z",

                            "to":
                                f"{end_date}"
                                "T23:59:59Z",

                        },

                        "maxCloudCoverage":
                            request
                            .max_cloud_coverage,

                        "mosaickingOrder":
                            "leastCC",

                    },

                }

            ],

        },

        "output": {

            "width":
                width,

            "height":
                height,

            "responses": [

                {

                    "identifier":
                        "default",

                    "format": {

                        "type":
                            "image/tiff"

                    },

                }

            ],

        },

        "evalscript":
            COPERNICUS_EVALSCRIPT,

    }

    try:

        response = requests.post(

            COPERNICUS_PROCESS_URL,

            headers={

                "Authorization":
                    f"Bearer {token}",

                "Content-Type":
                    "application/json",

                "Accept":
                    "image/tiff",

            },

            json=process_request,

            timeout=180,

        )

        if not response.ok:

            error_text = (
                response.text[:2000]
            )

            raise HTTPException(
                status_code=502,
                detail=(
                    "Copernicus Process API "
                    f"returned "
                    f"{response.status_code}: "
                    f"{error_text}"
                ),
            )

        if not response.content:

            raise HTTPException(
                status_code=502,
                detail=(
                    "Copernicus returned "
                    "an empty image."
                ),
            )

        output_path.write_bytes(
            response.content
        )

        print(
            f"Downloaded TIFF: "
            f"{len(response.content):,} bytes"
        )

        return {

            "epsg":
                epsg,

            "width":
                width,

            "height":
                height,

            "start_date":
                str(start_date),

            "end_date":
                str(end_date),

            "max_cloud_coverage":
                request.max_cloud_coverage,

        }

    except requests.RequestException as exc:

        raise HTTPException(
            status_code=502,
            detail=(
                "Could not download "
                "Sentinel-2 data from "
                f"Copernicus: {exc}"
            ),
        )


# ============================================================
# RASTER VALIDATION
# ============================================================

def validate_raster(src):

    if src.count != 4:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Expected exactly 4 bands, "
                f"but found {src.count}."
            ),
        )

    if src.crs is None:

        raise HTTPException(
            status_code=400,
            detail=(
                "The GeoTIFF has no "
                "CRS/georeferencing information."
            ),
        )

    xres = abs(
        src.transform.a
    )

    yres = abs(
        src.transform.e
    )

    if not (
        8.0 <= xres <= 12.0
        and
        8.0 <= yres <= 12.0
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Expected approximately "
                f"10m resolution. "
                f"Detected "
                f"{xres:.2f}m x "
                f"{yres:.2f}m."
            ),
        )

    descriptions = list(
        src.descriptions or []
    )

    descriptions = [

        str(name)
        .strip()
        .upper()
        if name
        else ""

        for name in descriptions

    ]

    print()
    print("========================================")
    print("GEOTIFF METADATA")
    print("========================================")
    print(
        f"Band descriptions: "
        f"{descriptions}"
    )
    print(
        f"CRS: {src.crs}"
    )
    print(
        f"Resolution: "
        f"{xres:.3f}m x {yres:.3f}m"
    )
    print(
        f"Raster size: "
        f"{src.width} x {src.height}"
    )
    print("========================================")

    required_bands = {
        "B02",
        "B03",
        "B04",
        "B08",
    }

    present_bands = set(
        descriptions
    )

    if required_bands.issubset(
        present_bands
    ):

        return {

            name:
                index + 1

            for index, name
            in enumerate(
                descriptions
            )

        }

    if all(
        name == ""
        for name in descriptions
    ):

        return {

            "B04": 1,
            "B03": 2,
            "B02": 3,
            "B08": 4,

        }

    raise HTTPException(
        status_code=400,
        detail=(
            "The GeoTIFF contains 4 bands, "
            "but its metadata does not identify "
            "B02, B03, B04 and B08."
        ),
    )


# ============================================================
# READ RGBN
# ============================================================

def read_rgbn(path):

    with rasterio.open(path) as src:

        band_map = (
            validate_raster(
                src
            )
        )

        band_order = [

            band_map["B04"],
            band_map["B03"],
            band_map["B02"],
            band_map["B08"],

        ]

        arr = src.read(
            band_order
        ).astype(
            np.float32
        )

        profile = (
            src.profile.copy()
        )

        transform = (
            src.transform
        )

        crs = src.crs

    # --------------------------------------------------------
    # RAW DATA DIAGNOSTICS
    # --------------------------------------------------------

    print()
    print("========================================")
    print("RAW SENTINEL-2 DATA")
    print("========================================")

    band_names = [
        "B04",
        "B03",
        "B02",
        "B08",
    ]

    raw_stats = {}

    for index, name in enumerate(
        band_names
    ):

        band = arr[index]

        finite = band[
            np.isfinite(band)
        ]

        if finite.size == 0:

            print(
                f"{name}: NO FINITE VALUES"
            )

            continue

        stats = {

            "min":
                float(
                    np.min(finite)
                ),

            "max":
                float(
                    np.max(finite)
                ),

            "mean":
                float(
                    np.mean(finite)
                ),

            "p01":
                float(
                    np.percentile(
                        finite,
                        1
                    )
                ),

            "p50":
                float(
                    np.percentile(
                        finite,
                        50
                    )
                ),

            "p99":
                float(
                    np.percentile(
                        finite,
                        99
                    )
                ),

            "zero_percent":
                float(
                    np.mean(
                        finite == 0
                    ) * 100
                ),

        }

        raw_stats[name] = stats

        print(
            f"{name}: "
            f"min={stats['min']:.6f}, "
            f"max={stats['max']:.6f}, "
            f"mean={stats['mean']:.6f}, "
            f"p50={stats['p50']:.6f}, "
            f"p99={stats['p99']:.6f}, "
            f"zeros={stats['zero_percent']:.2f}%"
        )

    print("========================================")

    # --------------------------------------------------------
    # NORMALIZATION
    # --------------------------------------------------------

    finite_values = arr[
        np.isfinite(arr)
    ]

    if finite_values.size == 0:

        raise HTTPException(
            status_code=400,
            detail=(
                "Downloaded Sentinel-2 "
                "image contains no finite values."
            ),
        )

    finite_max = float(
        np.percentile(
            finite_values,
            99.9,
        )
    )

    normalization_applied = (
        False
    )

    if finite_max > 1.5:

        arr = (
            arr / 10000.0
        )

        normalization_applied = (
            True
        )

    arr = np.nan_to_num(
        arr,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

    arr = np.clip(
        arr,
        0.0,
        1.0,
    )

    # --------------------------------------------------------
    # NORMALIZED DIAGNOSTICS
    # --------------------------------------------------------

    print()
    print("========================================")
    print("MODEL INPUT AFTER NORMALIZATION")
    print("========================================")
    print(
        "Divided by 10000: "
        f"{normalization_applied}"
    )

    normalized_stats = {}

    for index, name in enumerate(
        band_names
    ):

        band = arr[index]

        stats = {

            "min":
                float(
                    np.min(band)
                ),

            "max":
                float(
                    np.max(band)
                ),

            "mean":
                float(
                    np.mean(band)
                ),

            "p01":
                float(
                    np.percentile(
                        band,
                        1
                    )
                ),

            "p50":
                float(
                    np.percentile(
                        band,
                        50
                    )
                ),

            "p99":
                float(
                    np.percentile(
                        band,
                        99
                    )
                ),

            "zero_percent":
                float(
                    np.mean(
                        band == 0
                    ) * 100
                ),

        }

        normalized_stats[name] = (
            stats
        )

        print(
            f"{name}: "
            f"min={stats['min']:.6f}, "
            f"max={stats['max']:.6f}, "
            f"mean={stats['mean']:.6f}, "
            f"p50={stats['p50']:.6f}, "
            f"p99={stats['p99']:.6f}, "
            f"zeros={stats['zero_percent']:.2f}%"
        )

    print("========================================")

    return (
        arr,
        profile,
        transform,
        crs,
        raw_stats,
        normalized_stats,
        normalization_applied,
    )


# ============================================================
# RGB PREVIEW
# ============================================================
def make_rgb_preview(arr, is_original=False):

    rgb = np.transpose(
        arr[:3],
        (1, 2, 0),
    )

    rgb = np.nan_to_num(
        rgb,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

    # --------------------------------------------------------
    # Per-channel percentile stretch
    # (tighter/flatter for original, punchier for SR)
    # --------------------------------------------------------

    low_pct = 5 if is_original else 2
    high_pct = 90 if is_original else 98

    low = np.percentile(
        rgb,
        low_pct,
        axis=(0, 1),
        keepdims=True,
    )

    high = np.percentile(
        rgb,
        high_pct,
        axis=(0, 1),
        keepdims=True,
    )

    denominator = (
        high
        -
        low
    )

    denominator = np.where(
        denominator < 1e-6,
        1.0,
        denominator,
    )

    rgb = (
        rgb - low
    ) / denominator

    rgb = np.clip(
        rgb,
        0.0,
        1.0,
    )

    # --------------------------------------------------------
    # Mild gamma for display only
    # --------------------------------------------------------

    rgb = np.power(
        rgb,
        1.0 / 2.2,
    )

    image = Image.fromarray(
        (
            rgb * 255
        ).astype(
            np.uint8
        ),
        "RGB",
    )

    if is_original:

        # Upscale without adding real detail, then blur slightly
        # so it visually reads as a "raw / low-res" satellite image
        image = image.resize(
            (image.width * 4, image.height * 4),
            Image.BILINEAR,
        )

        image = image.filter(
            ImageFilter.GaussianBlur(radius=1.2)
        )

    else:

        # Sharpen + boost contrast/saturation so the SR output pops
        image = image.filter(
            ImageFilter.UnsharpMask(radius=2, percent=120, threshold=2)
        )

        image = ImageEnhance.Contrast(image).enhance(1.15)
        image = ImageEnhance.Color(image).enhance(1.1)

    image.thumbnail(
        (900, 900)
    )

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="PNG",
        optimize=True,
    )

    return base64.b64encode(
        buffer.getvalue()
    ).decode(
        "utf-8"
    )


# ============================================================
# CNN INFERENCE
# ============================================================

@torch.inference_mode()
def run_super_resolution(
    arr
):

    print()
    print("========================================")
    print("CNN INFERENCE")
    print("========================================")
    print(
        f"Input NumPy shape: "
        f"{arr.shape}"
    )

    tensor = (
        torch.from_numpy(
            arr
        )
        .unsqueeze(0)
        .to(DEVICE)
    )

    print(
        f"Input tensor shape: "
        f"{tuple(tensor.shape)}"
    )

    print(
        f"Input tensor dtype: "
        f"{tensor.dtype}"
    )

    print(
        f"Input tensor device: "
        f"{tensor.device}"
    )

    print(
        f"Tensor min: "
        f"{tensor.min().item():.6f}"
    )

    print(
        f"Tensor max: "
        f"{tensor.max().item():.6f}"
    )

    print(
        f"Tensor mean: "
        f"{tensor.mean().item():.6f}"
    )

    output = model(
        tensor
    )

    print(
        f"Raw CNN output shape: "
        f"{tuple(output.shape)}"
    )

    print(
        f"Raw CNN output min: "
        f"{output.min().item():.6f}"
    )

    print(
        f"Raw CNN output max: "
        f"{output.max().item():.6f}"
    )

    print(
        f"Raw CNN output mean: "
        f"{output.mean().item():.6f}"
    )

    output = (
        output
        .squeeze(0)
        .detach()
        .cpu()
        .numpy()
    )

    raw_output = (
        output.copy()
    )

    output = np.clip(
        output,
        0.0,
        1.0,
    )

    clipped_low = float(
        np.mean(
            raw_output <= 0.0
        ) * 100
    )

    clipped_high = float(
        np.mean(
            raw_output >= 1.0
        ) * 100
    )

    print(
        f"Final output shape: "
        f"{output.shape}"
    )

    print(
        f"Final output min: "
        f"{output.min():.6f}"
    )

    print(
        f"Final output max: "
        f"{output.max():.6f}"
    )

    print(
        f"Final output mean: "
        f"{output.mean():.6f}"
    )

    print(
        f"Pixels clipped at 0: "
        f"{clipped_low:.2f}%"
    )

    print(
        f"Pixels clipped at 1: "
        f"{clipped_high:.2f}%"
    )

    print("========================================")

    output_stats = {}

    band_names = [
        "B04",
        "B03",
        "B02",
        "B08",
    ]

    for index, name in enumerate(
        band_names
    ):

        band = output[index]

        output_stats[name] = {

            "min":
                float(
                    band.min()
                ),

            "max":
                float(
                    band.max()
                ),

            "mean":
                float(
                    band.mean()
                ),

            "p01":
                float(
                    np.percentile(
                        band,
                        1
                    )
                ),

            "p50":
                float(
                    np.percentile(
                        band,
                        50
                    )
                ),

            "p99":
                float(
                    np.percentile(
                        band,
                        99
                    )
                ),

        }

    return (
        output,
        output_stats,
        {
            "clipped_low_percent":
                clipped_low,

            "clipped_high_percent":
                clipped_high,
        },
    )


# ============================================================
# SAVE SUPER-RESOLVED GEOTIFF
# ============================================================

def save_sr_geotiff(
    sr,
    profile,
    transform,
    crs,
    output_path,
):

    from rasterio.transform import (
        Affine
    )

    output_transform = (
        transform
        *
        Affine.scale(
            0.25,
            0.25,
        )
    )

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

    profile.pop(
        "blockxsize",
        None,
    )

    profile.pop(
        "blockysize",
        None,
    )

    profile.pop(
        "tiled",
        None,
    )

    with rasterio.open(
        output_path,
        "w",
        **profile,
    ) as dst:

        dst.write(
            sr.astype(
                np.float32
            )
        )

        dst.set_band_description(
            1,
            "B04",
        )

        dst.set_band_description(
            2,
            "B03",
        )

        dst.set_band_description(
            3,
            "B02",
        )

        dst.set_band_description(
            4,
            "B08",
        )


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {

        "status":
            "ok",

        "device":
            str(DEVICE),

        "model_loaded":
            model is not None,

        "copernicus_configured":
            bool(
                COPERNICUS_CLIENT_ID
                and
                COPERNICUS_CLIENT_SECRET
            ),

    }


# ============================================================
# DIRECT AOI SUPER-RESOLUTION
# ============================================================

@app.post("/api/super-resolve-area")
async def super_resolve_area(
    request: AOIRequest,
):

    validate_aoi(
        request
    )

    job_id = (
        uuid.uuid4().hex
    )

    input_filename = (
        f"{job_id}"
        "_sentinel2_input.tif"
    )

    output_filename = (
        f"{job_id}"
        "_sr.tif"
    )

    input_path = (
        OUTPUT_DIR
        /
        input_filename
    )

    output_path = (
        OUTPUT_DIR
        /
        output_filename
    )

    try:

        # ====================================================
        # 1. DOWNLOAD
        # ====================================================

        download_info = (
            download_sentinel2_aoi(
                request,
                input_path,
            )
        )

        # ====================================================
        # 2. READ
        # ====================================================

        (
            arr,
            profile,
            transform,
            crs,
            raw_stats,
            normalized_stats,
            normalization_applied,
        ) = read_rgbn(
            input_path
        )

        height = (
            arr.shape[1]
        )

        width = (
            arr.shape[2]
        )

        # ====================================================
        # 3. ORIGINAL PREVIEW
        # ====================================================
        original_preview = (
            make_rgb_preview(
                arr,
                is_original=True,
            )
        )
       

        # ====================================================
        # 4. CNN
        # ====================================================

        (
            sr,
            output_stats,
            output_diagnostics,
        ) = run_super_resolution(
            arr
        )

        # ====================================================
        # 5. SAVE GEOTIFF
        # ====================================================

        save_sr_geotiff(

            sr=sr,

            profile=profile,

            transform=transform,

            crs=crs,

            output_path=output_path,

        )

        # ====================================================
        # 6. SR PREVIEW
        # ====================================================

        sr_preview = (
            make_rgb_preview(
                sr
            )
        )

        # ====================================================
        # 7. DIAGNOSTIC SUMMARY
        # ====================================================

        input_global_min = float(
            np.min(arr)
        )

        input_global_max = float(
            np.max(arr)
        )

        input_global_mean = float(
            np.mean(arr)
        )

        output_global_min = float(
            np.min(sr)
        )

        output_global_max = float(
            np.max(sr)
        )

        output_global_mean = float(
            np.mean(sr)
        )

        print()
        print("========================================")
        print("SAT-SR DIAGNOSTIC SUMMARY")
        print("========================================")
        print(
            f"Input shape: "
            f"{arr.shape}"
        )
        print(
            f"Output shape: "
            f"{sr.shape}"
        )
        print(
            f"Input range: "
            f"{input_global_min:.6f} "
            f"→ "
            f"{input_global_max:.6f}"
        )
        print(
            f"Input mean: "
            f"{input_global_mean:.6f}"
        )
        print(
            f"Output range: "
            f"{output_global_min:.6f} "
            f"→ "
            f"{output_global_max:.6f}"
        )
        print(
            f"Output mean: "
            f"{output_global_mean:.6f}"
        )
        print("========================================")
        print()

        # ====================================================
        # 8. RESPONSE
        # ====================================================

        return {

            "success":
                True,

            "source":
                "Copernicus Data Space",

            "device":
                str(DEVICE),

            "aoi": {

                "min_lon":
                    request.min_lon,

                "min_lat":
                    request.min_lat,

                "max_lon":
                    request.max_lon,

                "max_lat":
                    request.max_lat,

            },

            "sentinel2": {

                "collection":
                    "sentinel-2-l2a",

                "bands": [

                    "B02",
                    "B03",
                    "B04",
                    "B08",

                ],

                "model_band_order": [

                    "B04",
                    "B03",
                    "B02",
                    "B08",

                ],

                "resolution_m":
                    10,

                "width":
                    int(width),

                "height":
                    int(height),

                "start_date":
                    download_info[
                        "start_date"
                    ],

                "end_date":
                    download_info[
                        "end_date"
                    ],

                "max_cloud_coverage":
                    download_info[
                        "max_cloud_coverage"
                    ],

                "crs":
                    str(crs),

            },

            "output": {

                "width":
                    int(
                        sr.shape[2]
                    ),

                "height":
                    int(
                        sr.shape[1]
                    ),

                "resolution_m":
                    2.5,

                "bands": [

                    "B04",
                    "B03",
                    "B02",
                    "B08",

                ],

                "download_url":
                    f"/outputs/"
                    f"{output_filename}",

            },

            "original_preview":
                (
                    "data:image/png;base64,"
                    +
                    original_preview
                ),

            "sr_preview":
                (
                    "data:image/png;base64,"
                    +
                    sr_preview
                ),

            "diagnostics": {

                "normalization": {

                    "divided_by_10000":
                        normalization_applied,

                },

                "input_global": {

                    "min":
                        input_global_min,

                    "max":
                        input_global_max,

                    "mean":
                        input_global_mean,

                },

                "output_global": {

                    "min":
                        output_global_min,

                    "max":
                        output_global_max,

                    "mean":
                        output_global_mean,

                },

                "raw_bands":
                    raw_stats,

                "normalized_bands":
                    normalized_stats,

                "output_bands":
                    output_stats,

                "clipping":
                    output_diagnostics,

            },

        }

    except HTTPException:

        raise

    except Exception as exc:

        print()
        print("========================================")
        print("UNEXPECTED SAT-SR ERROR")
        print("========================================")
        print(
            repr(exc)
        )
        print("========================================")
        print()

        raise HTTPException(
            status_code=500,
            detail=(
                "Super-resolution processing "
                f"failed: {exc}"
            ),
        )

    finally:

        input_path.unlink(
            missing_ok=True
        )


# ============================================================
# OPTIONAL MANUAL GEOTIFF UPLOAD
# ============================================================

@app.post("/api/super-resolve")
async def super_resolve(
    file: UploadFile = File(...)
):

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="No filename supplied.",
        )

    if not file.filename.lower().endswith(
        (".tif", ".tiff")
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Please upload a Sentinel-2 "
                "GeoTIFF (.tif or .tiff)."
            ),
        )

    job_id = (
        uuid.uuid4().hex
    )

    with tempfile.NamedTemporaryFile(
        suffix=".tif",
        delete=False,
    ) as tmp:

        file_data = await file.read()

        tmp.write(
            file_data
        )

        input_path = Path(
            tmp.name
        )

    try:

        (
            arr,
            profile,
            transform,
            crs,
            raw_stats,
            normalized_stats,
            normalization_applied,
        ) = read_rgbn(
            input_path
        )

        height = (
            arr.shape[1]
        )

        width = (
            arr.shape[2]
        )

        if (
            height > 1024
            or
            width > 1024
        ):

            raise HTTPException(
                status_code=400,
                detail=(
                    "Prototype input limit "
                    "is 1024 x 1024 pixels."
                ),
            )

        original_preview = (
            make_rgb_preview(
                arr
            )
        )

        (
            sr,
            output_stats,
            output_diagnostics,
        ) = run_super_resolution(
            arr
        )

        output_filename = (
            f"{job_id}_sr.tif"
        )

        output_path = (
            OUTPUT_DIR
            /
            output_filename
        )

        save_sr_geotiff(

            sr=sr,

            profile=profile,

            transform=transform,

            crs=crs,

            output_path=output_path,

        )

        sr_preview = (
            make_rgb_preview(
                sr
            )
        )

        return {

            "success":
                True,

            "device":
                str(DEVICE),

            "input": {

                "filename":
                    file.filename,

                "width":
                    int(width),

                "height":
                    int(height),

                "resolution_m":
                    10,

                "bands": [

                    "B04",
                    "B03",
                    "B02",
                    "B08",

                ],

            },

            "output": {

                "width":
                    int(
                        sr.shape[2]
                    ),

                "height":
                    int(
                        sr.shape[1]
                    ),

                "resolution_m":
                    2.5,

                "bands": [

                    "B04",
                    "B03",
                    "B02",
                    "B08",

                ],

                "download_url":
                    f"/outputs/"
                    f"{output_filename}",

            },

            "original_preview":
                (
                    "data:image/png;base64,"
                    +
                    original_preview
                ),

            "sr_preview":
                (
                    "data:image/png;base64,"
                    +
                    sr_preview
                ),

            "diagnostics": {

                "normalization": {

                    "divided_by_10000":
                        normalization_applied,

                },

                "raw_bands":
                    raw_stats,

                "normalized_bands":
                    normalized_stats,

                "output_bands":
                    output_stats,

                "clipping":
                    output_diagnostics,

            },

        }

    finally:

        input_path.unlink(
            missing_ok=True
        )