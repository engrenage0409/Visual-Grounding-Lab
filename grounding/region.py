"""Patch-anchored candidate boxes and supervision for V5 region grounding.

Each CLIP patch predicts a text-conditioned score, center offset and box size.
Unlike spatial expectation, inference selects a scored candidate instead of
averaging locations of unrelated objects.
"""
import math
import torch
from torch.nn import functional as F
from .geometry import box_iou_aligned, generalized_iou_aligned


def region_candidates(features, score_head, offset_head, size_head, grid):
    """features: [B, grid*grid, D]; returns logits and normalized xyxy boxes."""
    batch, n, _ = features.shape
    if n != grid * grid:
        raise ValueError(f"Expected {grid*grid} region features, got {n}")
    coords = (torch.arange(grid, device=features.device, dtype=features.dtype) + .5) / grid
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    anchor = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
    center = (anchor[None] + torch.tanh(offset_head(features)) * (.5 / grid)).clamp(0, 1)
    size = torch.sigmoid(size_head(features))
    half = size / 2
    boxes = torch.cat([(center - half).clamp(0, 1), (center + half).clamp(0, 1)], dim=-1)
    scores = score_head(features).squeeze(-1)
    return scores, boxes


def region_positive_indices(target, grid):
    """Grid cell containing GT *amodal* center, shape [B]."""
    cx = (target[:, 0] + target[:, 2]) / 2
    cy = (target[:, 1] + target[:, 3]) / 2
    col = (cx * grid).long().clamp(0, grid-1)
    row = (cy * grid).long().clamp(0, grid-1)
    return row * grid + col


def region_grounding_loss(pred, target, aux, loss_type="giou", region_weight=1.0):
    """Train selector on GT center cell and box regressor at that same cell.

    Returned IoU measures the actual selected inference box, not the positive
    training candidate; train/val metrics therefore remain comparable.
    """
    logits = aux["region_logits"]
    candidates = aux["region_boxes"]
    grid = math.isqrt(logits.shape[1])
    if grid * grid != logits.shape[1] or candidates.shape != (*logits.shape, 4):
        raise ValueError("Region logits/boxes must have matching square grid")
    positive = region_positive_indices(target, grid)
    batch_idx = torch.arange(len(positive), device=target.device)
    pos_boxes = candidates[batch_idx, positive]
    if loss_type == "giou":
        geometric = (1 - generalized_iou_aligned(pos_boxes, target)).mean()
    elif loss_type == "iou":
        geometric = (1 - box_iou_aligned(pos_boxes, target)).mean()
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
    localization = F.l1_loss(pos_boxes, target) + geometric
    selection = F.cross_entropy(logits, positive)
    loss = localization + region_weight * selection
    return loss, box_iou_aligned(pred, target)
