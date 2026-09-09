import numpy as np
import matplotlib.pyplot as plt


def make_rgb(image: np.ndarray) -> np.ndarray:
    """
    Convert a 4-band RGBN image in B04, B03, B02, B08 order
    into an RGB image for visualization.
    """

    rgb = image[:3].transpose(1, 2, 0)

    # Robust contrast stretch for display.
    low = np.percentile(rgb, 2)
    high = np.percentile(rgb, 98)

    rgb = (rgb - low) / (high - low + 1e-8)
    rgb = np.clip(rgb, 0.0, 1.0)

    return rgb


def create_comparison(
    original: np.ndarray,
    super_resolved: np.ndarray,
):
    """
    Create a side-by-side comparison of the original
    10 m image and the 2.5 m super-resolved image.
    """

    original_rgb = make_rgb(original)
    sr_rgb = make_rgb(super_resolved)

    figure, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].imshow(original_rgb)
    axes[0].set_title("Original Sentinel-2 — 10 m")
    axes[0].axis("off")

    axes[1].imshow(sr_rgb)
    axes[1].set_title("SUP_RESO — 2.5 m")
    axes[1].axis("off")

    figure.tight_layout()

    return figure