# Visual Grounding reading list

1. [CLIP: Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) — image/text feature alignment and ViT patch representation.
2. [REFER API / RefCOCO / RefCOCO+ / RefCOCOg](https://github.com/lichengunc/refer) — referring-expression datasets and split definitions.
3. [RefCOCO on TFDS](https://www.tensorflow.org/datasets/catalog/ref_coco) — annotation fields and manual image download requirements.
4. [Grounding DINO paper](https://arxiv.org/abs/2303.05499) and [official implementation](https://github.com/IDEA-Research/GroundingDINO) — open-set text-conditioned object detection and comparison baseline.
5. [Segment Anything 2](https://github.com/facebookresearch/sam2) — possible future mask segmentation extension; not implemented here.

## Study checklist

- Image + referring-expression pair and pixel-space xyxy labels.
- CLIP ViT patch tokens, pooled text representation, attention Q/K/V.
- Bounding box regression, normalized coordinates, aligned IoU and L1+IoU loss.
- Accuracy@IoU 0.5; per-image visualizations and failure analysis.
- Compare trained box head with pretrained Grounding DINO.
- Later: Grounding DINO + SAM, industrial drawings + OCR + geometry.

This project is **not** a reproduction of Grounding DINO or a published benchmark result.
