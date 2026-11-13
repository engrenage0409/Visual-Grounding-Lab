"""V5.1 FCOS-inspired anchor-free referring-expression head.

Each text-conditioned patch predicts L/T/R/B distances from a reference point.
A center-sampled set of cells inside the target box is supervised (not only one
center cell). Scores combine a target-match logit and predicted box-quality
logit. This is intentionally a lightweight single-scale research baseline.
"""
import torch
import torch.nn.functional as F
from .geometry import box_iou_aligned, generalized_iou_aligned


def grid_points(grid, device, dtype):
    coords = (torch.arange(grid, device=device, dtype=dtype) + .5) / grid
    y, x = torch.meshgrid(coords, coords, indexing="ij")
    return torch.stack([x.reshape(-1), y.reshape(-1)], dim=-1)


def anchor_free_candidates(features, classification_head, distance_head,
                           quality_head, grid):
    batch, n, _ = features.shape
    if n != grid * grid:
        raise ValueError(f"Expected {grid*grid} cells, got {n}")
    points = grid_points(grid, features.device, features.dtype)
    # Distances are normalized to image size, not constrained to one patch.
    distances = torch.sigmoid(distance_head(features))
    x = points[None, :, 0]
    y = points[None, :, 1]
    l, t, r, b = distances.unbind(-1)
    boxes = torch.stack([(x-l).clamp(0, 1), (y-t).clamp(0, 1),
                         (x+r).clamp(0, 1), (y+b).clamp(0, 1)], dim=-1)
    class_logits = classification_head(features).squeeze(-1)
    quality_logits = quality_head(features).squeeze(-1)
    scores = torch.sigmoid(class_logits) * torch.sigmoid(quality_logits)
    return class_logits, quality_logits, boxes, scores


def positive_point_mask(target, grid, radius=1.5):
    """Interior points near GT center; fallback to nearest cell if target tiny."""
    points = grid_points(grid, target.device, target.dtype)
    x, y = points[:, 0][None], points[:, 1][None]
    inside = ((x >= target[:, 0:1]) & (x <= target[:, 2:3]) &
              (y >= target[:, 1:2]) & (y <= target[:, 3:4]))
    cx = (target[:, 0:1] + target[:, 2:3]) * .5
    cy = (target[:, 1:2] + target[:, 3:4]) * .5
    near = ((x-cx).abs() <= radius / grid) & ((y-cy).abs() <= radius / grid)
    mask = inside & near
    absent = ~mask.any(dim=1)
    if absent.any():
        dist = (x-cx).square() + (y-cy).square()
        idx = dist.argmin(dim=1)
        mask[absent, idx[absent]] = True
    return mask


def anchor_free_loss(pred, target, aux, loss_type="giou", region_weight=1.0):
    """Multi-positive classification + localization + quality estimation."""
    cls = aux["af_class_logits"]
    quality = aux["af_quality_logits"]
    boxes = aux["region_boxes"]
    grid = int(cls.shape[1] ** .5)
    if grid*grid != cls.shape[1] or boxes.shape != (*cls.shape, 4):
        raise ValueError("Expected matching square-grid anchor-free predictions")
    pos = positive_point_mask(target, grid)
    target_flags = pos.to(cls.dtype)
    # Give positives and negatives comparable total weight per image.
    bce = F.binary_cross_entropy_with_logits(cls, target_flags, reduction="none")
    pos_loss = (bce * target_flags).sum(-1) / target_flags.sum(-1).clamp(min=1)
    neg_flags = 1 - target_flags
    neg_loss = (bce * neg_flags).sum(-1) / neg_flags.sum(-1).clamp(min=1)
    classification = (pos_loss + neg_loss).mean()

    expanded_gt = target[:, None, :].expand_as(boxes)
    positive_boxes = boxes[pos]
    positive_gt = expanded_gt[pos]
    if loss_type == "giou":
        geometric = (1 - generalized_iou_aligned(positive_boxes, positive_gt)).mean()
    elif loss_type == "iou":
        geometric = (1 - box_iou_aligned(positive_boxes, positive_gt)).mean()
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
    localization = F.l1_loss(positive_boxes, positive_gt) + geometric

    # Soft quality = detached IoU of positive candidate, zero elsewhere.
    with torch.no_grad():
        target_quality = torch.zeros_like(quality)
        target_quality[pos] = box_iou_aligned(positive_boxes.detach(), positive_gt)
    quality_bce = F.binary_cross_entropy_with_logits(quality, target_quality,
                                                      reduction="none")
    quality_loss = ((quality_bce * target_flags).sum(-1) /
                    target_flags.sum(-1).clamp(min=1) +
                    (quality_bce * neg_flags).sum(-1) /
                    neg_flags.sum(-1).clamp(min=1)).mean()
    loss = localization + region_weight * (classification + .5 * quality_loss)
    return loss, box_iou_aligned(pred, target)
