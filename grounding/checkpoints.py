"""Load both old baseline checkpoints and new architecture-aware head checkpoints."""
import torch
from .model import CLIPGrounder

def load_grounder(path,device="cpu"):
    ck=torch.load(path,map_location="cpu",weights_only=True)
    arch=ck.get("arch","baseline")
    model=CLIPGrounder(ck["model_name"],hidden=ck.get("hidden",256),arch=arch).to(device)
    missing,unexpected=model.load_state_dict(ck["head"],strict=False)
    # Old checkpoints intentionally omit all frozen CLIP weights.
    permitted=[key for key in missing if key.startswith("clip.")]
    if len(permitted)!=len(missing) or unexpected:
        raise ValueError(f"Checkpoint incompatible: missing={missing}, unexpected={unexpected}")
    model.eval()
    return model,ck
