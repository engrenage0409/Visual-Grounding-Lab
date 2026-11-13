# V2: Text-conditioned spatial localization

## Motivation
On the 300-image Mixed synthetic dataset, the Baseline run reported:
- Epoch 1: train loss 1.1293, validation mean IoU 0.0813, Acc@0.5 0/60
- Epoch 5: train loss 1.1148, validation mean IoU 0.0752, Acc@0.5 3/60
- For the same image, three text queries yielded nearly identical central boxes.

These are **user-provided local-run results**, not a controlled final benchmark.
The new spatial variant is intended to test whether explicit patch supervision
prevents the approximately text-invariant / central-location output.

## V2 mechanism
1. Use frozen CLIP ViT-B/32 patch features (7×7), frozen CLIP text features.
2. Project both to 256 hidden dimensions.
3. Add *learned positional features* to 49 patch tokens.
4. Fuse text with each patch separately and compute 49 logits.
5. Softmax to form text-conditioned spatial probability map.
6. Expected patch center (x,y) defines box center; pooled features predict (w,h).
7. Optimize normalized XYXY L1 + (1-GIoU) + spatial cross-entropy.
   Spatial soft labels are proportional to **ground-truth box / patch overlap area**.

It is **not** a segmentation mask: only box-derived pseudo spatial supervision.
The patch grid is coarse and head has no high-resolution feature pyramid.

## Reproducible comparison
- Baseline: old `checkpoints/mixed.pt` from prior run. Old checkpoint loading stays supported.
- V2: `--arch spatial --loss giou --out checkpoints/mixed_spatial.pt`.
- For ablations: `--arch baseline --loss giou` isolates loss change,
  `--arch spatial --loss iou` isolates spatial head effect.
- Evaluate on same mixed validation; keep hard validation separate.
- Compare a single image queried with different textual references and inspect numerical boxes and spatial maps.
- Beware small training sample count and synthetic-only data.

Full tests and GPU training must be run in the target environment before claiming improvement.
