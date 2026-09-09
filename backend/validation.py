from pathlib import Path

import rasterio


REQUIRED_BANDS = {"B02", "B03", "B04", "B08"}
EXPECTED_RESOLUTION = 10.0


class ValidationError(Exception):
    """Raised when an uploaded satellite image is incompatible with SUP_RESO."""


def validate_satellite_image(file_path: str) -> dict:
    """
    Validate a Sentinel-2 10 m RGBN GeoTIFF before inference.

    Required bands:
        B02 - Blue
        B03 - Green
        B04 - Red
        B08 - NIR

    Returns metadata required by the inference pipeline.
    """

    path = Path(file_path)

    if not path.exists():
        raise ValidationError("Input file does not exist.")

    if path.suffix.lower() not in {".tif", ".tiff"}:
        raise ValidationError("Only GeoTIFF files (.tif/.tiff) are supported.")

    try:
        with rasterio.open(path) as src:

            # 1. Must contain exactly four bands.
            if src.count != 4:
                raise ValidationError(
                    f"Expected exactly 4 bands, but found {src.count}."
                )

            # 2. Check spatial resolution.
            resolution_x = abs(src.transform.a)
            resolution_y = abs(src.transform.e)

            if (
                abs(resolution_x - EXPECTED_RESOLUTION) > 0.01
                or abs(resolution_y - EXPECTED_RESOLUTION) > 0.01
            ):
                raise ValidationError(
                    f"Expected 10 m resolution, but found "
                    f"{resolution_x:.2f} m × {resolution_y:.2f} m."
                )

            # 3. Try to identify Sentinel-2 band names from metadata.
            band_names = []

            for band_index in range(1, src.count + 1):
                description = src.descriptions[band_index - 1]

                if description:
                    band_names.append(description.upper().strip())
                else:
                    band_names.append("")

            metadata_bands = set(band_names)

            # If band descriptions are available, require the exact bands.
            if all(band_names):
                if metadata_bands != REQUIRED_BANDS:
                    raise ValidationError(
                        "The four bands must be B02, B03, B04 and B08."
                    )

            # 4. Check that the raster can actually be read.
            data = src.read()

            if data.size == 0:
                raise ValidationError("The raster contains no pixel data.")

            # 5. Check for completely invalid data.
            if not data.any():
                raise ValidationError("The raster contains no valid pixel values.")

            return {
                "path": str(path),
                "width": src.width,
                "height": src.height,
                "band_count": src.count,
                "resolution_x": resolution_x,
                "resolution_y": resolution_y,
                "band_names": band_names,
                "crs": str(src.crs) if src.crs else None,
            }

    except ValidationError:
        raise

    except Exception as exc:
        raise ValidationError(
            f"Unable to read the satellite raster: {exc}"
        ) from exc