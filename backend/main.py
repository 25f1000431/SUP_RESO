from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from pathlib import Path
import tempfile
import uuid
import base64
import io

import numpy as np
import torch
import rasterio
from PIL import Image

from sen2sr.models.opensr_baseline.cnn import CNNSR


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "sen2sr_full_2851roi.pth"

OUTPUT_DIR = BASE_DIR / "backend" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="SAT-SR",
    description="Sentinel-2 Super Resolution Prototype",
    version="1.0.0",
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
# STATIC OUTPUT FILES
# ============================================================

app.mount(
    "/outputs",
    StaticFiles(directory=str(OUTPUT_DIR)),
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
            f"Model checkpoint not found: {MODEL_PATH}"
        )

    print("Loading SAT-SR model...")
    print(f"Checkpoint: {MODEL_PATH}")
    print(f"Device: {DEVICE}")

    # --------------------------------------------------------
    # EXACT ARCHITECTURE USED FOR OUR FINE-TUNED MODEL
    # --------------------------------------------------------

    model = CNNSR(
        in_channels=4,
        out_channels=4,
        feature_channels=24,
        upscale=4,
        bias=True,
        train_mode=False,
        num_blocks=6,
    )

    # --------------------------------------------------------
    # LOAD CHECKPOINT
    # --------------------------------------------------------

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    # Our checkpoint contains "model_state_dict".
    # This also allows loading a raw state_dict if needed.
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.to(DEVICE)

    model.eval()

    print("========================================")
    print("SAT-SR MODEL LOADED")
    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_PATH}")
    print("========================================")


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup_event():

    load_model()


# ============================================================
# SENTINEL-2 INPUT VALIDATION
# ============================================================
def validate_raster(src):

    # --------------------------------------------------------
    # EXACTLY 4 BANDS
    # --------------------------------------------------------

    if src.count != 4:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Expected exactly 4 bands, "
                f"but found {src.count}."
            ),
        )

    # --------------------------------------------------------
    # CRS
    # --------------------------------------------------------

    if src.crs is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "The uploaded GeoTIFF has no CRS/"
                "georeferencing information."
            ),
        )

    # --------------------------------------------------------
    # RESOLUTION
    # --------------------------------------------------------

    xres = abs(src.transform.a)
    yres = abs(src.transform.e)

    if not (
        8.0 <= xres <= 12.0
        and
        8.0 <= yres <= 12.0
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Expected approximately 10m resolution. "
                f"Detected {xres:.2f}m × {yres:.2f}m."
            ),
        )

    # --------------------------------------------------------
    # BAND METADATA
    # --------------------------------------------------------

    descriptions = list(
        src.descriptions or []
    )

    descriptions = [
        str(name).strip().upper()
        if name
        else ""
        for name in descriptions
    ]

    required_bands = {
        "B02",
        "B03",
        "B04",
        "B08",
    }

    present_bands = set(
        descriptions
    )

    # --------------------------------------------------------
    # CASE 1:
    # TIFF HAS BAND NAMES
    # --------------------------------------------------------

    if required_bands.issubset(
        present_bands
    ):

        return {
            name: index + 1
            for index, name in enumerate(
                descriptions
            )
        }

    # --------------------------------------------------------
    # CASE 2:
    # TIFF HAS NO BAND NAMES
    #
    # SEN2NAIP LR files can be 4-band TIFFs without
    # B02/B03/B04/B08 descriptions.
    #
    # For our prototype, we know the four channels are:
    #
    # 1 = B04
    # 2 = B03
    # 3 = B02
    # 4 = B08
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # PARTIAL / UNKNOWN BAND METADATA
    # --------------------------------------------------------

    raise HTTPException(
        status_code=400,
        detail=(
            "The GeoTIFF contains 4 bands, but its "
            "band metadata does not identify B02, B03, "
            "B04 and B08."
        ),
    )


# ============================================================
# READ SENTINEL-2 RGBN
# ============================================================

def read_rgbn(path):

    with rasterio.open(path) as src:

        band_map = validate_raster(src)

        # ----------------------------------------------------
        # MODEL BAND ORDER
        #
        # B04 = Red
        # B03 = Green
        # B02 = Blue
        # B08 = NIR
        # ----------------------------------------------------

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

        profile = src.profile.copy()

        transform = src.transform

        crs = src.crs

    # --------------------------------------------------------
    # NORMALIZATION
    #
    # Sentinel-2 reflectance is commonly stored as 0–10000.
    # Our model expects approximately 0–1.
    # --------------------------------------------------------

    finite_max = float(
        np.nanpercentile(
            arr,
            99.9,
        )
    )

    if finite_max > 1.5:
        arr = arr / 10000.0

    # Remove invalid numerical values
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

    return (
        arr,
        profile,
        transform,
        crs,
    )


# ============================================================
# CREATE RGB PREVIEW
# ============================================================

def make_rgb_preview(arr):

    # Input is already:
    #
    # [B04, B03, B02, B08]
    #
    # So first three channels are RGB.

    rgb = np.transpose(
        arr[:3],
        (1, 2, 0),
    )

    # --------------------------------------------------------
    # DISPLAY STRETCH
    # This is ONLY for visualization.
    # It does not modify model inference.
    # --------------------------------------------------------

    low = np.percentile(
        rgb,
        2,
        axis=(0, 1),
        keepdims=True,
    )

    high = np.percentile(
        rgb,
        98,
        axis=(0, 1),
        keepdims=True,
    )

    rgb = (
        rgb - low
    ) / (
        high - low + 1e-6
    )

    rgb = np.clip(
        rgb,
        0.0,
        1.0,
    )

    image = Image.fromarray(
        (rgb * 255).astype(
            np.uint8
        ),
        "RGB",
    )

    # Keep browser response reasonably small
    image.thumbnail(
        (900, 900)
    )

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="PNG",
        optimize=True,
    )

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode(
        "utf-8"
    )

    return encoded


# ============================================================
# SUPER-RESOLUTION INFERENCE
# ============================================================

@torch.inference_mode()
def run_super_resolution(arr):

    # --------------------------------------------------------
    # NumPy:
    #
    # [4, H, W]
    #
    # PyTorch:
    #
    # [1, 4, H, W]
    # --------------------------------------------------------

    tensor = torch.from_numpy(
        arr
    ).unsqueeze(
        0
    )

    tensor = tensor.to(
        DEVICE
    )

    # --------------------------------------------------------
    # CNN INFERENCE
    # --------------------------------------------------------

    output = model(
        tensor
    )

    # [1, 4, H*4, W*4]
    # →
    # [4, H*4, W*4]

    output = output.squeeze(
        0
    )

    output = output.detach().cpu().numpy()

    output = np.clip(
        output,
        0.0,
        1.0,
    )

    return output


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

    from rasterio.transform import Affine

    # --------------------------------------------------------
    # 4× spatial resolution
    #
    # 10m → 2.5m
    # --------------------------------------------------------

    output_transform = (
        transform
        * Affine.scale(
            0.25,
            0.25,
        )
    )

    # --------------------------------------------------------
    # UPDATE GEOTIFF PROFILE
    # --------------------------------------------------------

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

    # Remove potentially incompatible input tiling
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

    # --------------------------------------------------------
    # WRITE OUTPUT
    # --------------------------------------------------------

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
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "device": str(DEVICE),
        "model_loaded": model is not None,
    }


# ============================================================
# SUPER-RESOLUTION API
# ============================================================

@app.post("/api/super-resolve")
async def super_resolve(
    file: UploadFile = File(...)
):

    # --------------------------------------------------------
    # FILE TYPE CHECK
    # --------------------------------------------------------

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

    job_id = uuid.uuid4().hex

    # --------------------------------------------------------
    # SAVE TEMPORARY INPUT
    # --------------------------------------------------------

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

        # ----------------------------------------------------
        # READ + VALIDATE
        # ----------------------------------------------------

        (
            arr,
            profile,
            transform,
            crs,
        ) = read_rgbn(
            input_path
        )

        height = arr.shape[1]
        width = arr.shape[2]

        # ----------------------------------------------------
        # PROTOTYPE SIZE LIMIT
        # ----------------------------------------------------

        if (
            height > 1024
            or
            width > 1024
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    "Prototype input limit is "
                    "1024 × 1024 pixels. "
                    "Please upload a smaller Sentinel-2 tile."
                ),
            )

        # ----------------------------------------------------
        # ORIGINAL IMAGE PREVIEW
        # ----------------------------------------------------

        original_preview = (
            make_rgb_preview(
                arr
            )
        )

        # ----------------------------------------------------
        # AI SUPER-RESOLUTION
        # ----------------------------------------------------

        sr = run_super_resolution(
            arr
        )

        # ----------------------------------------------------
        # SAVE SR GEOTIFF
        # ----------------------------------------------------

        output_filename = (
            f"{job_id}_sr.tif"
        )

        output_path = (
            OUTPUT_DIR
            / output_filename
        )

        save_sr_geotiff(
            sr=sr,
            profile=profile,
            transform=transform,
            crs=crs,
            output_path=output_path,
        )

        # ----------------------------------------------------
        # SR PREVIEW
        # ----------------------------------------------------

        sr_preview = make_rgb_preview(
            sr
        )

        # ----------------------------------------------------
        # RETURN RESULT TO REACT
        # ----------------------------------------------------

        return {
            "success": True,

            "input": {
                "filename": file.filename,
                "width": int(width),
                "height": int(height),
                "resolution_m": 10,
                "bands": [
                    "B04",
                    "B03",
                    "B02",
                    "B08",
                ],
            },

            "output": {
                "width": int(
                    sr.shape[2]
                ),
                "height": int(
                    sr.shape[1]
                ),
                "resolution_m": 2.5,
                "bands": [
                    "B04",
                    "B03",
                    "B02",
                    "B08",
                ],
                "download_url": (
                    f"/outputs/"
                    f"{output_filename}"
                ),
            },

            "device": str(
                DEVICE
            ),

            "original_preview": (
                "data:image/png;base64,"
                + original_preview
            ),

            "sr_preview": (
                "data:image/png;base64,"
                + sr_preview
            ),
        }

    finally:

        # ----------------------------------------------------
        # DELETE TEMPORARY INPUT
        # ----------------------------------------------------

        input_path.unlink(
            missing_ok=True
        )