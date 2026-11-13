"""Load a saved V7 head without altering V1-V6 checkpoint handling."""
import torch
from .generative import GenerativeGrounder


def load_generative(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("arch") != "generative" or checkpoint.get("coord_max", 1000) not in (100,224,1000):
        raise ValueError("Expected V7 generative checkpoint with valid coordinate vocabulary")
    model = GenerativeGrounder(
        model_name=checkpoint["model_name"],
        hidden=checkpoint["hidden"],
        layers=checkpoint["layers"],
        heads=checkpoint["heads"],
        dropout=checkpoint["dropout"],
        coord_max=checkpoint.get("coord_max",1000)).to(device)
    missing, unexpected = model.load_state_dict(checkpoint["head"], strict=False)
    if any(not key.startswith("clip.") for key in missing) or unexpected:
        raise ValueError(f"Incompatible V7 head: missing={missing}, unexpected={unexpected}")
    model.eval()
    return model, checkpoint
