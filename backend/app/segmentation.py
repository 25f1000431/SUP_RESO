from pathlib import Path
from PIL import Image
import numpy as np
import rasterio
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp


BASE_DIR = Path(__file__).resolve().parents[2]

CHECKPOINT_PATH = (
    BASE_DIR / "models" / "deeplabv3plus_final_crop_health.pt"
)


# IMPORTANT:
# This is the ACTUAL order used while training the model.
# The checkpoint metadata says B2/B3/B4/B8, but the training
# notebook loaded Sentinel-2 as B4/B3/B2/B8.
MODEL_BAND_ORDER = ["B04", "B03", "B02", "B08"]


CLASS_NAMES = {
    0: "background",
    1: "cropland",
    2: "landslide",
}


def ndvi_to_health_score(ndvi):
    """
    Relative NDVI-based health score used in the
    crop-health notebook.

    NDVI <= 0.2 -> 0
    NDVI >= 0.8 -> 100
    """

    score = ((ndvi - 0.2) / 0.6) * 100

    return float(np.clip(score, 0, 100))


def health_label(score):

    if score < 20:
        return "Very Low"

    elif score < 40:
        return "Low"

    elif score < 60:
        return "Moderate"

    elif score < 80:
        return "High"

    else:
        return "Very High"


def create_segmentation_preview(mask: np.ndarray) -> Image.Image:
    """
    Create a browser-friendly RGB preview from the segmentation mask.

    Classes:
        0 = background
        1 = cropland
        2 = landslide

    This only creates a visualization.
    The original segmentation GeoTIFF is not modified.
    """

    h, w = mask.shape

    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    # Background
    rgb[mask == 0] = [25, 30, 27]

    # Cropland
    rgb[mask == 1] = [70, 170, 90]

    # Landslide
    rgb[mask == 2] = [190, 110, 70]

    return Image.fromarray(rgb, mode="RGB")


def create_health_preview(
    ndvi: np.ndarray,
    crop_mask: np.ndarray,
) -> Image.Image:
    """
    Create a spatial Crop Health Index visualization.

    Health score:
        NDVI <= 0.2  -> 0
        NDVI >= 0.8  -> 100

    Non-cropland pixels are shown in white.
    """

    h, w = ndvi.shape

    rgb = np.full(
        (h, w, 3),
        255,
        dtype=np.uint8,
    )

    valid = (
        (crop_mask == 1)
        & np.isfinite(ndvi)
    )

    if not np.any(valid):
        return Image.fromarray(
            rgb,
            mode="RGB",
        )

    # Same health formula already used
    # for the overall crop health score.
    health = (
        (ndvi - 0.2) / 0.6
    ) * 100.0

    health = np.clip(
        health,
        0,
        100,
    )

    values = health[valid] / 100.0

    # 0   = red
    # 50  = yellow
    # 100 = green

    r = np.zeros_like(values)
    g = np.zeros_like(values)
    b = np.zeros_like(values)

    low = values < 0.5
    high = ~low

    # Red -> Yellow
    r[low] = 255
    g[low] = 255 * (values[low] * 2)

    # Yellow -> Green
    r[high] = 255 * (1 - (values[high] - 0.5) * 2)
    g[high] = 255

    rgb[valid, 0] = np.clip(r, 0, 255).astype(np.uint8)
    rgb[valid, 1] = np.clip(g, 0, 255).astype(np.uint8)
    rgb[valid, 2] = np.clip(b, 0, 255).astype(np.uint8)

    return Image.fromarray(
        rgb,
        mode="RGB",
    )


def create_ndvi_preview(
    ndvi: np.ndarray,
    crop_mask: np.ndarray,
) -> Image.Image:
    """
    Create a browser-friendly crop NDVI visualization.

    Only pixels classified as cropland are visualized.
    Non-cropland pixels remain dark.

    This is ONLY a visualization.
    The original NDVI GeoTIFF is not modified.
    """

    h, w = ndvi.shape

    rgb = np.zeros(
        (h, w, 3),
        dtype=np.uint8,
    )

    # Dark background for non-crop pixels.
    rgb[:] = [15, 20, 17]

    valid = (
        (crop_mask == 1)
        & np.isfinite(ndvi)
    )

    if not np.any(valid):
        return Image.fromarray(
            rgb,
            mode="RGB",
        )

    # Display-only NDVI normalization.
    #
    # NDVI:
    # -0.2 → low display value
    #  0.8 → high display value
    #
    # This does NOT change the actual NDVI GeoTIFF.
    normalized = np.clip(
        (ndvi + 0.2) / 1.0,
        0,
        1,
    )

    values = normalized[valid]

    # Low NDVI → yellow/brown
    # Medium NDVI → green
    # High NDVI → bright green

    r = np.clip(
        255 * (1.0 - values),
        0,
        255,
    )

    g = np.clip(
        80 + 175 * values,
        0,
        255,
    )

    b = np.clip(
        70 * (1.0 - values),
        0,
        255,
    )

    rgb[valid, 0] = r.astype(np.uint8)
    rgb[valid, 1] = g.astype(np.uint8)
    rgb[valid, 2] = b.astype(np.uint8)

    return Image.fromarray(
        rgb,
        mode="RGB",
    )

class DeepLabSegmenter:

    def __init__(self, device="cpu"):

        self.device = torch.device(device)

        print("Loading DeepLabV3+...")

        self.model = smp.DeepLabV3Plus(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=4,
            classes=3,
        )

        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=self.device,
            weights_only=False,
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True,
        )

        self.model.to(self.device)
        self.model.eval()

        print("DeepLabV3+ loaded successfully.")

    def predict(
        self,
        sr_path: str,
        output_path: str,
    ):

        # =====================================================
        # READ SR DATA
        # =====================================================

        print(f"Reading SR image: {sr_path}")

        with rasterio.open(sr_path) as src:

            image = src.read().astype(np.float32)

            profile = src.profile.copy()

            transform = src.transform

        if image.shape[0] != 4:

            raise ValueError(
                f"Expected 4-band SR image, "
                f"got {image.shape[0]} bands"
            )

        print("SR image shape:", image.shape)

        print(
            "SR value range:",
            float(np.nanmin(image)),
            "to",
            float(np.nanmax(image)),
        )

        print(
            "Using DeepLab band order:",
            MODEL_BAND_ORDER,
        )

        # =====================================================
        # IMPORTANT:
        #
        # SEN2SR output is already:
        #
        # B04 / B03 / B02 / B08
        #
        # which matches the ACTUAL DeepLab training order.
        #
        # DO NOT reorder here.
        # DO NOT divide by 10000 again.
        # =====================================================
        # =====================================================
        # PREPARE INPUT FOR DEEPLAB
        # =====================================================

        model_image = image.copy()

        # Original Sentinel-2 data may be stored as
        # digital numbers (typically 0-10000).
        # Convert to reflectance when necessary.

        if np.nanmax(model_image) > 1.5:
            model_image = model_image / 10000.0

        model_image = np.clip(
            model_image,
            0.0,
            1.0,
        ).astype(np.float32)
        

        # =====================================================
        # CREATE MODEL TENSOR
        # =====================================================

        tensor = (
            torch.from_numpy(model_image)
            .unsqueeze(0)
            .to(self.device)
        )

        original_h = tensor.shape[-2]
        original_w = tensor.shape[-1]

        # =====================================================
        # PAD TO MULTIPLE OF 16
        # =====================================================

        pad_h = (
            16 - original_h % 16
        ) % 16

        pad_w = (
            16 - original_w % 16
        ) % 16

        if pad_h > 0 or pad_w > 0:

            print(
                f"Padding image: "
                f"{original_h}x{original_w} -> "
                f"{original_h + pad_h}x"
                f"{original_w + pad_w}"
            )

            tensor = F.pad(
                tensor,
                (0, pad_w, 0, pad_h),
                mode="reflect",
            )

        # =====================================================
        # DEEPLAB INFERENCE
        # =====================================================

        print(
            "Running DeepLabV3+ inference..."
        )

        with torch.no_grad():

            output = self.model(tensor)

        # Remove padding

        output = output[
            :,
            :,
            :original_h,
            :original_w,
        ]

        # =====================================================
        # CLASS PREDICTION
        # =====================================================

        prediction = torch.argmax(
            output,
            dim=1,
        )

        mask = (
            prediction
            .squeeze(0)
            .cpu()
            .numpy()
            .astype(np.uint8)
        )

        print(
            "Segmentation mask shape:",
            mask.shape,
        )

        print(
            "Detected classes:",
            np.unique(mask),
        )

        # =====================================================
        # SAVE SEGMENTATION MASK
        # =====================================================

        profile.update(
            count=1,
            dtype="uint8",
            nodata=255,
        )

        with rasterio.open(
            output_path,
            "w",
            **profile,
        ) as dst:

            dst.write(mask, 1)

        print(
            f"Segmentation saved: {output_path}"
        )

        # =====================================================
        # SAVE SEGMENTATION PREVIEW
        # =====================================================

        segmentation_preview_path = str(
            Path(output_path).with_name(
                Path(output_path).stem
                + "_preview.png"
            )
        )

        segmentation_preview = create_segmentation_preview(
            mask
        )

        segmentation_preview.save(
            segmentation_preview_path,
            format="PNG",
        )

        print(
            f"Segmentation preview saved: "
            f"{segmentation_preview_path}"
        )

        # =====================================================
        # CLASS STATISTICS
        # =====================================================

        total_pixels = mask.size

        statistics = {}

        for class_id, class_name in CLASS_NAMES.items():

            count = int(
                np.sum(mask == class_id)
            )

            percentage = (
                count / total_pixels * 100
                if total_pixels > 0
                else 0
            )

            statistics[class_name] = {
                "pixels": count,
                "percentage": round(
                    percentage,
                    2,
                ),
            }

        # =====================================================
        # PIXEL AREA
        # =====================================================

        pixel_width = abs(
            transform.a
        )

        pixel_height = abs(
            transform.e
        )

        pixel_area_m2 = (
            pixel_width *
            pixel_height
        )

        for class_name in statistics:

            pixel_count = statistics[
                class_name
            ]["pixels"]

            area_m2 = (
                pixel_count *
                pixel_area_m2
            )

            statistics[class_name][
                "area_m2"
            ] = round(
                area_m2,
                2,
            )

            statistics[class_name][
                "area_hectares"
            ] = round(
                area_m2 / 10000,
                4,
            )

        # =====================================================
        # NDVI
        #
        # ACTUAL MODEL/SR ORDER:
        #
        # channel 0 = B04 RED
        # channel 1 = B03 GREEN
        # channel 2 = B02 BLUE
        # channel 3 = B08 NIR
        # =====================================================

        red = image[0]

        nir = image[3]

        denominator = (
            nir + red
        )

        ndvi = np.divide(
            nir - red,
            denominator,
            out=np.zeros_like(red),
            where=np.abs(
                denominator
            ) > 1e-8,
        )

        ndvi = np.clip(
            ndvi,
            -1,
            1,
        )

        # =====================================================
        # CROPLAND NDVI
        # =====================================================

        crop_mask = (
            mask == 1
        )

        crop_ndvi = ndvi[
            crop_mask
        ]

        crop_health = {
            "available": False,
            "mean_ndvi": None,
            "median_ndvi": None,
            "min_ndvi": None,
            "max_ndvi": None,
            "std_ndvi": None,
            "health_score": None,
            "health_label": "No Cropland",
        }

        # =====================================================
        # HEALTH CALCULATION
        # =====================================================

        if crop_ndvi.size > 0:

            mean_ndvi = float(
                np.mean(crop_ndvi)
            )

            median_ndvi = float(
                np.median(crop_ndvi)
            )

            min_ndvi = float(
                np.min(crop_ndvi)
            )

            max_ndvi = float(
                np.max(crop_ndvi)
            )

            std_ndvi = float(
                np.std(crop_ndvi)
            )

            health_score = (
                ndvi_to_health_score(
                    mean_ndvi
                )
            )

            crop_health = {
                "available": True,

                "mean_ndvi": round(
                    mean_ndvi,
                    4,
                ),

                "median_ndvi": round(
                    median_ndvi,
                    4,
                ),

                "min_ndvi": round(
                    min_ndvi,
                    4,
                ),

                "max_ndvi": round(
                    max_ndvi,
                    4,
                ),

                "std_ndvi": round(
                    std_ndvi,
                    4,
                ),

                "health_score": round(
                    health_score,
                    1,
                ),

                "health_label": health_label(
                    health_score
                ),
            }

        # =====================================================
        # VIGOR DISTRIBUTION
        # =====================================================

        vigor_distribution = {
            "very_low": 0,
            "low": 0,
            "moderate": 0,
            "high": 0,
            "very_high": 0,
        }

        if crop_ndvi.size > 0:

            total_crop = crop_ndvi.size

            vigor_distribution = {

                "very_low": round(
                    np.sum(
                        crop_ndvi < 0.2
                    )
                    / total_crop
                    * 100,
                    2,
                ),

                "low": round(
                    np.sum(
                        (crop_ndvi >= 0.2)
                        & (crop_ndvi < 0.4)
                    )
                    / total_crop
                    * 100,
                    2,
                ),

                "moderate": round(
                    np.sum(
                        (crop_ndvi >= 0.4)
                        & (crop_ndvi < 0.6)
                    )
                    / total_crop
                    * 100,
                    2,
                ),

                "high": round(
                    np.sum(
                        (crop_ndvi >= 0.6)
                        & (crop_ndvi < 0.8)
                    )
                    / total_crop
                    * 100,
                    2,
                ),

                "very_high": round(
                    np.sum(
                        crop_ndvi >= 0.8
                    )
                    / total_crop
                    * 100,
                    2,
                ),
            }

        # =====================================================
        # SAVE CROPLAND-ONLY NDVI MAP
        # =====================================================

        ndvi_output_path = str(
            Path(output_path).with_name(
                Path(output_path).stem
                + "_crop_ndvi.tif"
            )
        )

        crop_ndvi_map = np.full_like(
            ndvi,
            np.nan,
            dtype=np.float32,
        )

        crop_ndvi_map[
            crop_mask
        ] = ndvi[
            crop_mask
        ]

        ndvi_profile = profile.copy()

        ndvi_profile.update(
            count=1,
            dtype="float32",
        )

        with rasterio.open(
            ndvi_output_path,
            "w",
            **ndvi_profile,
        ) as dst:

            dst.write(
                crop_ndvi_map,
                1,
            )
        # =====================================================
        # SAVE NDVI / CROP HEALTH PREVIEWS
        # =====================================================

        # -----------------------------------------------------
        # NDVI preview
        # -----------------------------------------------------

        ndvi_preview_path = str(
            Path(output_path).with_name(
                Path(output_path).stem
                + "_crop_ndvi_preview.png"
            )
        )

        ndvi_preview = create_ndvi_preview(
            ndvi,
            crop_mask,
        )

        ndvi_preview.save(
            ndvi_preview_path,
            format="PNG",
        )

        print(
            f"NDVI preview saved: "
            f"{ndvi_preview_path}"
        )


        # -----------------------------------------------------
        # Crop Health Index preview
        # -----------------------------------------------------

        health_preview_path = str(
            Path(output_path).with_name(
                Path(output_path).stem
                + "_crop_health_preview.png"
            )
        )

        health_preview = create_health_preview(
            ndvi,
            crop_mask,
        )

        health_preview.save(
            health_preview_path,
            format="PNG",
        )

        print(
            f"Crop health preview saved: "
            f"{health_preview_path}"
        )

 


        # =====================================================
        # SAVE CROP HEALTH INDEX PREVIEW
        # =====================================================

        health_preview_path = str(
            Path(output_path).with_name(
                Path(output_path).stem
                + "_crop_health_preview.png"
            )
        )

        health_preview = create_health_preview(
            ndvi,
            crop_mask,
        )

        health_preview.save(
            health_preview_path,
            format="PNG",
        )

        print(
            f"Crop health preview saved: "
            f"{health_preview_path}"
        )

        print(
            f"NDVI preview saved: "
            f"{ndvi_preview_path}"
        )    

        # =====================================================
        # RETURN STRUCTURED RESULT
        # =====================================================
        return {

            "mask_path": output_path,

            "ndvi_path": ndvi_output_path,

            "segmentation_preview_path": segmentation_preview_path,

            "ndvi_preview_path": ndvi_preview_path,

            "health_preview_path": health_preview_path,

            "shape": list(
                mask.shape
            ),

            "classes": statistics,

            "crop_health": crop_health,

            "vigor_distribution": (
                vigor_distribution
            ),

            "pixel_size_m": {
                "x": pixel_width,
                "y": pixel_height,
            },

            "bands": MODEL_BAND_ORDER,
        }