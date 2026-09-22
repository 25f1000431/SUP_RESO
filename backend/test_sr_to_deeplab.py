import os

import numpy as np
import rasterio
import torch
import segmentation_models_pytorch as smp


SR_PATH = "outputs/1fecdf1a26034adeacbc46de9bde764a_sr.tif"
CHECKPOINT = "../models/deeplabv3plus_final_crop_health.pt"
OUTPUT_PATH = "outputs/1fecdf1a26034adeacbc46de9bde764a_segmentation.tif"


print("Loading SR image...")

with rasterio.open(SR_PATH) as src:
    image = src.read().astype(np.float32)
    profile = src.profile.copy()

print("SR shape:", image.shape)
print("SR dtype:", image.dtype)
print("SR min:", np.nanmin(image))
print("SR max:", np.nanmax(image))

# ---------------------------------------------------------
# DeepLabV3+ model
# ---------------------------------------------------------

print("\nCreating DeepLabV3+...")

model = smp.DeepLabV3Plus(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=4,
    classes=3,
)

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=False,
)

state_dict = checkpoint["model_state_dict"]

model.load_state_dict(state_dict, strict=True)

model.eval()

print("Checkpoint loaded.")

# ---------------------------------------------------------
# Convert HWC -> BCHW
# ---------------------------------------------------------

tensor = torch.from_numpy(image).unsqueeze(0)

print("Model input:", tuple(tensor.shape))

# ---------------------------------------------------------
# Inference
# ---------------------------------------------------------

print("\nRunning DeepLabV3+...")

with torch.no_grad():
    output = model(tensor)

print("Raw output:", tuple(output.shape))

# Class with highest probability/logit
prediction = torch.argmax(output, dim=1)

mask = prediction.squeeze(0).cpu().numpy().astype(np.uint8)

print("Mask shape:", mask.shape)
print("Unique classes:", np.unique(mask))

# ---------------------------------------------------------
# Save segmentation mask
# ---------------------------------------------------------

profile.update(
    count=1,
    dtype="uint8",
    nodata=255,
)

with rasterio.open(OUTPUT_PATH, "w", **profile) as dst:
    dst.write(mask, 1)

print("\nSegmentation saved:")
print(OUTPUT_PATH)

# ---------------------------------------------------------
# Class statistics
# ---------------------------------------------------------

total_pixels = mask.size

background = np.sum(mask == 0)
cropland = np.sum(mask == 1)
landslide = np.sum(mask == 2)

print("\nSegmentation statistics:")
print(f"Background: {background} pixels ({background / total_pixels * 100:.2f}%)")
print(f"Cropland:   {cropland} pixels ({cropland / total_pixels * 100:.2f}%)")
print(f"Landslide:  {landslide} pixels ({landslide / total_pixels * 100:.2f}%)")