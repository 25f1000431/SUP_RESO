"""Runs the loaded CNN on a normalized RGBN array."""

import numpy as np
import torch

from app.config import DEVICE, BAND_NAMES
from app import model_loader


@torch.inference_mode()
def run_super_resolution(arr: np.ndarray):
    model = model_loader.require_model()

    tensor = torch.from_numpy(arr).unsqueeze(0).to(DEVICE)
    raw_output = model(tensor).squeeze(0).detach().cpu().numpy()

    clipped_low = float(np.mean(raw_output <= 0.0) * 100)
    clipped_high = float(np.mean(raw_output >= 1.0) * 100)
    output = np.clip(raw_output, 0.0, 1.0)

    print(f"[INFERENCE] output range {output.min():.4f}-{output.max():.4f}, "
          f"clipped_low={clipped_low:.2f}% clipped_high={clipped_high:.2f}%")

    output_stats = {
        name: {
            "min": float(output[i].min()),
            "max": float(output[i].max()),
            "mean": float(output[i].mean()),
            "p01": float(np.percentile(output[i], 1)),
            "p50": float(np.percentile(output[i], 50)),
            "p99": float(np.percentile(output[i], 99)),
        }
        for i, name in enumerate(BAND_NAMES)
    }

    clipping = {"clipped_low_percent": clipped_low, "clipped_high_percent": clipped_high}

    return output, output_stats, clipping
