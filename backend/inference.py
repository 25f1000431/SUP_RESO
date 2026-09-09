from pathlib import Path

import numpy as np
import rasterio
import torch

from sen2sr.models.opensr_baseline.cnn import CNNSR


MODEL_PATH = Path("models/sen2sr_finetuned.pth")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    """Load the fine-tuned SEN2SRLite CNN."""

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
    )

    # Support both a raw state_dict and a checkpoint dictionary.
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return model


def prepare_input(input_path: str) -> np.ndarray:
    """
    Read a validated Sentinel-2 RGBN GeoTIFF.

    Output band order:
        B04, B03, B02, B08

    Output values are normalized to approximately [0, 1].
    """

    with rasterio.open(input_path) as src:
        data = src.read().astype(np.float32)

        descriptions = [
            (description or "").upper().strip()
            for description in src.descriptions
        ]

        # If metadata contains band names, reorder automatically.
        if all(descriptions):
            band_map = {
                name: index
                for index, name in enumerate(descriptions)
            }

            required = ["B04", "B03", "B02", "B08"]

            if all(band in band_map for band in required):
                data = np.stack(
                    [data[band_map[band]] for band in required],
                    axis=0,
                )

        # Sentinel-2 reflectance is commonly stored as DN / 10000.
        data /= 10000.0

        data = np.clip(data, 0.0, 1.0)

    return data


@torch.inference_mode()
def run_inference(input_path: str) -> np.ndarray:
    """Run the fine-tuned CNN and return the 4x SR output."""

    model = load_model()

    data = prepare_input(input_path)

    tensor = torch.from_numpy(data).unsqueeze(0).to(DEVICE)

    output = model(tensor)

    output = output.squeeze(0).cpu().numpy()

    output = np.clip(output, 0.0, 1.0)

    return output