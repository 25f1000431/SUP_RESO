import torch
import segmentation_models_pytorch as smp


CHECKPOINT = "../models/deeplabv3plus_final_crop_health.pt"


print("Creating DeepLabV3+...")

model = smp.DeepLabV3Plus(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=4,
    classes=3,
)

print("Model created.")

print("Loading checkpoint...")

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=False,
)

print("Checkpoint type:", type(checkpoint))

if isinstance(checkpoint, dict):
    print("Checkpoint keys:", list(checkpoint.keys())[:20])

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
else:
    state_dict = checkpoint

model.load_state_dict(state_dict, strict=True)

print("Checkpoint loaded successfully.")

# Test 4-band input
x = torch.randn(1, 4, 128, 128)

print("Running test inference...")

model.eval()

with torch.no_grad():
    output = model(x)

print("Input shape :", tuple(x.shape))
print("Output shape:", tuple(output.shape))

print("\nDeepLabV3+ test successful.")