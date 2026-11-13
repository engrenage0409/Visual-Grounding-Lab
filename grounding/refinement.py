"""V6: multi-scale coarse-to-fine box refinement using frozen CLIP patch maps.

Uses two scales of the same final-layer ViT feature map (native and pooled);
it is not a true backbone FPN. Zero-initialized residual keeps V4 behavior at
initialization when loading a compatible V4 spatial checkpoint.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .geometry import xyxy_to_cxcywh
from .spatial import center_size_to_valid_xyxy


def roi_grid(boxes, samples=3):
    """Normalized xyxy boxes [B,4] -> grid_sample coordinates [B,S,S,2]."""
    u = torch.linspace(0, 1, samples, device=boxes.device, dtype=boxes.dtype)
    gy, gx = torch.meshgrid(u, u, indexing="ij")
    x = boxes[:, 0, None, None] + gx[None] * (boxes[:, 2] - boxes[:, 0])[:, None, None]
    y = boxes[:, 1, None, None] + gy[None] * (boxes[:, 3] - boxes[:, 1])[:, None, None]
    return torch.stack([2*x-1, 2*y-1], dim=-1)


def multiscale_roi_features(tokens, boxes, grid, samples=3):
    """ROI pooled feature from native and half-resolution CLIP token maps.

    tokens: [B, grid*grid, H], boxes: [B,4], output [B,2*H].
    """
    b, n, d = tokens.shape
    if n != grid*grid:
        raise ValueError(f"Expected {grid*grid} spatial tokens, got {n}")
    fmap = tokens.transpose(1, 2).reshape(b, d, grid, grid)
    coarse = F.adaptive_avg_pool2d(fmap, (max(1, grid//2), max(1, grid//2)))
    sample_grid = roi_grid(boxes, samples=samples)
    fine_roi = F.grid_sample(fmap, sample_grid, mode="bilinear",
                             padding_mode="border", align_corners=False).mean((-1,-2))
    coarse_roi = F.grid_sample(coarse, sample_grid, mode="bilinear",
                               padding_mode="border", align_corners=False).mean((-1,-2))
    return torch.cat([fine_roi, coarse_roi], dim=-1)


class CoarseFineRefiner(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.refine_head = nn.Sequential(
            nn.LayerNorm(hidden*3+4),
            nn.Linear(hidden*3+4, hidden),
            nn.GELU(),
            nn.Linear(hidden, 4))
        # Strong guarantee: loading V4 and enabling V6 initially changes no boxes.
        nn.init.zeros_(self.refine_head[-1].weight)
        nn.init.zeros_(self.refine_head[-1].bias)

    def forward(self, visual_tokens, text_query, coarse_boxes, grid):
        roi = multiscale_roi_features(visual_tokens, coarse_boxes, grid)
        input_features = torch.cat([roi, text_query, coarse_boxes], dim=-1)
        delta = torch.tanh(self.refine_head(input_features))
        # Bounded center shifts and multiplicative size refinements.
        center_size = xyxy_to_cxcywh(coarse_boxes)
        center = center_size[:, :2] + .15 * delta[:, :2]
        size = center_size[:, 2:].clamp(min=1e-5) * torch.exp(.5 * delta[:, 2:])
        refined = center_size_to_valid_xyxy(center.clamp(0,1), size.clamp(max=1))
        return refined, delta
