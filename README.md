# Visual Grounding Lab — Referring Expression → Bounding Box

Starter implementation of **trainable language-guided object localization** on an 8GB-class GPU, e.g. RTX 5060. Uses **frozen CLIP ViT-B/32 image patch and text representations**, plus trainable projections, cross-attention and one-box regression head. This is a learning baseline, **not** a trained RefCOCO benchmark or Grounding DINO reproduction.

## Current state

Implemented: three-level synthetic data generator (easy/mixed/hard), JSONL data loader, frozen CLIP visual/text feature extraction, baseline cross-attention and V2 spatially supervised heads, box regression, L1 + IoU/GIoU loss, evaluation (Acc@IoU≥0.5 and mean IoU), and visualizations. **No official real-data metrics, trained adapters, or SAM masks yet.** Synthetic data is solely for smoke testing. The generator now supports easy, mixed and hard scenes with randomized object positions, shapes, sizes, colors and controlled occlusion. Bounding boxes for occluded targets are full-object (amodal) boxes.

## Directory structure

```text
grounding/
  model.py           Baseline + V2 spatially supervised heads\n  spatial.py         7x7 patch-overlap supervision + spatial center\n  checkpoints.py     Load old and new head-only checkpoints
  geometry.py        box format transformations and aligned IoU
  data.py            image / text / xyxy JSONL data loader
  engine.py          losses and evaluation
scripts/
  make_toy_data.py   easy/mixed/hard shapes, randomized layouts and occlusion
  train.py           head training
  evaluate.py        validation
  visualize.py       prediction and optional 7x7 spatial heatmap
tests/
  test_geometry.py   geometry unit tests
docs/
  reading-list.md    links to papers, datasets and comparison baselines
```

## Installation

From the repository root, use an existing CUDA-enabled PyTorch environment (the one used for CILP is a reasonable starting point). If needed, install PyTorch from the [official selector](https://pytorch.org/get-started/locally/).

```powershell
git clone https://github.com/engrenage0409/Visual-Grounding-Lab.git
cd Visual-Grounding-Lab
python -m pip install -r requirements.txt
python -m pytest -q
```

Hugging Face downloads the public CLIP backbone on first run.

## Phase 0: sanity check (synthetic rectangles only)

Run every command from the repository root:

```powershell
python -m scripts.make_toy_data
python -m scripts.train --epochs 5 --batch-size 4
python -m scripts.evaluate
python -m scripts.visualize --image data/toy/val_0000.png --text "the red rectangle"
```

Output: `checkpoints/grounder.pt` (only trained head; frozen CLIP weights reload from Hugging Face), validation Acc@0.5 and mean IoU, and `results/grounding_preview.png`.

### Difficulty-controlled synthetic experiments

The original dataset placed the same colored rectangles at nearly fixed horizontal positions, encouraging a *position shortcut*. The updated generator uses a disjoint random stream for validation and three modes:

```powershell
python -m scripts.make_toy_data --mode easy  --out data/easy  --train 300 --val 60
python -m scripts.make_toy_data --mode mixed --out data/mixed --train 500 --val 100
python -m scripts.make_toy_data --mode hard  --out data/hard  --train 500 --val 100
```

- `easy`: randomizes left-to-right color order, retaining three rectangles as a control.
- `mixed`: varied colors, circle/triangle/rectangle shapes, scale, position; little overlap.
- `hard`: five to seven objects, same-color/same-shape distractors and partial overlap; target selected to preserve at least ~68% visible area.
- Labels remain `image`, `expression`, `bbox_xyxy` and include optional `objects`, `difficulty`, `target_visible_fraction`, `bbox_type=amodal`.

For independent comparisons, **train and evaluate on separate mode-specific files**. For example:

```powershell
python -m scripts.train --train data/mixed/train.jsonl --val data/mixed/val.jsonl --image-root data/mixed --epochs 5 --batch-size 4 --out checkpoints/mixed.pt
```

Because images use a synthetic renderer, split samples remain procedurally similar: these numbers do not measure generalization to real referring expressions. Hard-mode occluded boxes are **amodal**. The default synthetic benchmark should not be confused with official RefCOCO annotation protocols.

## Phase 0.5: V2 spatial localization (new)

The first local Mixed run (300 train / 60 validation) achieved **5% Acc@0.5 and 0.0752 mean IoU after 5 epochs**, with nearly identical boxes for different text prompts. This indicates the baseline is not learning effective language-conditioned localization under that setup.

**The original baseline model and checkpoints are preserved.** `--arch spatial` (now the training default) adds a text-conditioned 49-patch spatial probability map, box-overlap soft target supervision, learned patch position embeddings, and a separate width/height head. `--loss giou` adds GIoU to handle non-overlapping boxes. **No V2 validation results have been measured yet.**

After `git pull origin main`:

```powershell
python -m pytest -q
python -m scripts.train --arch spatial --loss giou --train data/mixed/train.jsonl --val data/mixed/val.jsonl --image-root data/mixed --epochs 10 --batch-size 4 --out checkpoints/mixed_spatial.pt
python -m scripts.evaluate --checkpoint checkpoints/mixed_spatial.pt --data data/mixed/val.jsonl --image-root data/mixed
python -m scripts.evaluate --checkpoint checkpoints/mixed_spatial.pt --data data/hard/val.jsonl --image-root data/hard
```

Create **one box image and one spatial heatmap**:

```powershell
python -m scripts.visualize --checkpoint checkpoints/mixed_spatial.pt --image data/mixed/val_0000.png --text "the purple circle" --output results/purple_spatial.png --heatmap results/purple_attention.png
```

To compare against the *unchanged* baseline, evaluate the prior `checkpoints/mixed.pt`; loaders automatically detect older checkpoints without `arch`. Ablation options: `--arch baseline --loss giou` and `--arch spatial --loss iou`. Keep checkpoints separate. See [V2 experimental design](docs/v2-experiment.md).

**Important:** Attention is a learned box-supervised spatial probability map, not an explanation guarantee or pixel-level segmentation mask. The target is box overlap with coarse 7x7 CLIP patches.

## Phase 1: actual referring-expression data

Obtain RefCOCO-style images and annotations legally, and split examples by the appropriate dataset protocol. Convert to per-line JSON records:

```json
{"image":"000000123.jpg","expression":"the person in a blue shirt","bbox_xyxy":[13,24,145,205]}
```

The box must be **xyxy in original image pixels**. The loader divides by original width and height; `image` is relative to `--image-root` unless absolute.

```powershell
python -m scripts.train --train data/refcoco/train.jsonl --val data/refcoco/val.jsonl --image-root data/refcoco/images --epochs 10 --batch-size 4
python -m scripts.evaluate --data data/refcoco/val.jsonl --image-root data/refcoco/images
```

No licensed RefCOCO data or automatic downloader is bundled. Follow original COCO and refer-expression dataset licensing and train/validation split definitions.

## Architecture

```text
Image ── frozen CLIP ViT ── patch embeddings ── projection ─┐
                                                            ├─ cross-attention ─ box MLP ─ normalized xyxy
Text  ── frozen CLIP text encoder ─ pooled vector ─ project ┘
```

Vision and text backbone parameters stay frozen for both architectures. The head emits **one box per expression**; it does not perform region proposal generation, multi-box open-set detection, mask prediction, or multi-scale grounding. CLIP ViT-B/32 has coarse patch spatial resolution, limiting small-object performance.

## Metrics and limits

- **Acc@0.5:** proportion of predicted boxes with IoU ≥ 0.5 against ground truth.
- **Mean IoU:** average overlap across examples.
- Keep test data untouched while selecting epochs and hyperparameters.
- Default training is a single fixed seed with no measured results in this repository yet; do not report synthetic smoke-test accuracy as a benchmark.
- Dependencies are not yet fully pinned to an exact reproduction environment.

## Next experiments

1. Official RefCOCO/RefCOCOg data adaptation and correct evaluation splits.
2. Error analysis by object size and referring-expression type.
3. Grounding DINO pretrained baseline.
4. Industrial drawing grounding with OCR and geometric evidence.
5. Optional SAM box-prompt segmentation.

See [papers and resources](docs/reading-list.md).
