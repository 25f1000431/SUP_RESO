"""Turns a (4, H, W) reflectance array into a display-ready PNG (base64)."""

import base64
import io

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def make_rgb_preview(arr: np.ndarray, is_original: bool = False) -> str:
    rgb = np.nan_to_num(np.transpose(arr[:3], (1, 2, 0)), nan=0.0, posinf=1.0, neginf=0.0)

    # Original images get a gentler stretch; SR output a punchier one.
    low_pct, high_pct = (5, 90) if is_original else (2, 98)
    low = np.percentile(rgb, low_pct, axis=(0, 1), keepdims=True)
    high = np.percentile(rgb, high_pct, axis=(0, 1), keepdims=True)
    denom = np.where((high - low) < 1e-6, 1.0, high - low)

    rgb = np.clip((rgb - low) / denom, 0.0, 1.0) ** (1.0 / 2.2)  # display gamma
    image = Image.fromarray((rgb * 255).astype(np.uint8), "RGB")

    if is_original:
        # Upscale + blur so it visually reads as "raw / low-res" imagery.
        image = image.resize((image.width * 4, image.height * 4), Image.BILINEAR)
        image = image.filter(ImageFilter.GaussianBlur(radius=1.2))
    else:
        image = image.filter(ImageFilter.UnsharpMask(radius=2, percent=120, threshold=2))
        image = ImageEnhance.Contrast(image).enhance(1.15)
        image = ImageEnhance.Color(image).enhance(1.1)

    image.thumbnail((900, 900))

    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def as_data_uri(base64_png: str) -> str:
    return f"data:image/png;base64,{base64_png}"
