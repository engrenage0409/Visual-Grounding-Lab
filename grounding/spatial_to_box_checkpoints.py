"""Load V7.2 head-only checkpoint against the same frozen V6 reference."""
import torch
from grounding.spatial_to_generative import SpatialToBox


def load_spatial_to_box(path,device="cpu",v6_checkpoint=None):
    ck=torch.load(path,map_location="cpu",weights_only=True)
    if ck.get("arch")!="spatial_to_box":
        raise ValueError("Not a V7.2 spatial_to_box checkpoint")
    pretrained_path=v6_checkpoint or ck["v6_checkpoint"]
    model=SpatialToBox(pretrained_path,mode=ck["mode"],hidden=ck["hidden"],
                       coord_max=ck["coord_max"],layers=ck["layers"],
                       heads=ck["heads"],dropout=ck["dropout"],device=device).to(device)
    if model.v6.model_name!=ck["v6_model_name"]:
        raise ValueError("V6 backbone does not match saved V7.2 head")
    missing,unexpected=model.load_state_dict(ck["head"],strict=False)
    if any(not key.startswith("v6.") for key in missing) or unexpected:
        raise ValueError(f"Checkpoint incompatible: {missing=}, {unexpected=}")
    model.eval()
    return model,ck
