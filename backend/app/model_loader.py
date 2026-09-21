"""
Loads the SAT-SR CNN checkpoint once at startup and exposes it to the
rest of the app. Loading never crashes the server -- if it fails, the
error is stored and endpoints that need the model return a clean 503.
"""

from fastapi import HTTPException
import torch

from sen2sr.models.opensr_baseline.cnn import CNNSR

from app.config import MODEL_PATH, DEVICE

model: CNNSR | None = None
load_error: str | None = None


def load_model() -> None:
    global model, load_error

    if not MODEL_PATH.exists():
        load_error = f"Model checkpoint not found: {MODEL_PATH}"
        print(f"[STARTUP WARNING] {load_error}")
        return

    print(f"[STARTUP] Loading SAT-SR model from {MODEL_PATH} on {DEVICE}")

    try:
        candidate = CNNSR(
            in_channels=4,
            out_channels=4,
            feature_channels=24,
            upscale=4,
            bias=True,
            train_mode=False,
            num_blocks=6,
        )

        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        state_dict = (
            checkpoint.get("model_state_dict")
            or checkpoint.get("state_dict")
            or checkpoint
        )

        candidate.load_state_dict(state_dict, strict=True)
        candidate.to(DEVICE).eval()

        model = candidate
        load_error = None
        print("[STARTUP] SAT-SR model loaded successfully.")

    except Exception as exc:
        load_error = f"{type(exc).__name__}: {exc}"
        print(f"[STARTUP ERROR] Failed to load model: {load_error}")


def require_model() -> CNNSR:
    """Raise a clear 503 if the model isn't ready, otherwise return it."""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Model is not loaded: {load_error or 'unknown error'}. "
                "Check backend startup logs and MODEL_PATH."
            ),
        )
    return model
