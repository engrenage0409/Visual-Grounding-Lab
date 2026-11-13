## V7.1: diagnose learning of coordinate tokens (experimental)

V7 was able to reduce training token CE from ~5.0 to ~3.15 but measured 0% Hard Acc@0.5. Since EOS is always the fifth target token, a perfect EOS predictor can attain 20% token accuracy **without locating anything**. V7.1 adds more informative diagnostics and coordinate-vocabulary ablations while **preserving V7's existing checkpoint and V1–V6 heads**.

**New measures:** train coordinate token accuracy (four coordinates only), train EOS accuracy (separately), *teacher-forced* validation box IoU (GT prior coordinate tokens supplied), actual **autoregressive** box IoU (prior predictions supplied), actual coordinate exact-match and normalized MAE, invalid box rate and EOS rate. Teacher-forced box metrics are diagnostics and **not real inference performance**. Note that GT coordinate tokens still provide an easier history and token argmax may be invalid. These metrics expose generation failure without confusing it with EOS memorization.

**Resolution ablation:** `--coord-max 100`, `224` or `1000` changes discrete coordinate vocabulary size (103 / 227 / 1003 tokens including BOS/EOS); each run starts a fresh decoder. Coordinate quantization limits absolute error to approximately 0.5 / coord_max per scalar (before learning errors). Old V7 checkpoints default to resolution 1000 on load. Do **not** load weights between different vocab sizes.

**Optional decoding constraint:** `--constrain-box` is a **decode-only** ablation: x1,y1 are restricted below the maximum, x2>x1 and y2>y1. This prevents invalid box geometry but cannot ensure correct target selection and must be compared against unconstrained decoding.

```powershell
git pull origin main
python -m pytest -q

# Test diagnostics on your original V7 (no retraining):
python -m scripts.evaluate_generative --checkpoint checkpoints/v7_generative_b16.pt --data data/v4_hard/val.jsonl --image-root data/v4_hard
python -m scripts.evaluate_generative --checkpoint checkpoints/v7_generative_b16.pt --data data/v4_hard/val.jsonl --image-root data/v4_hard --constrain-box

# Quantization ablation: identical data/architecture/training, independent saved weights:
python -m scripts.train_generative --coord-max 100 --train data/v4_joint/train.jsonl --val data/v4_mixed/val.jsonl --image-root data/v4_mixed --model openai/clip-vit-base-patch16 --epochs 10 --batch-size 4 --out checkpoints/v7_1_gen_q100.pt
python -m scripts.train_generative --coord-max 224 --train data/v4_joint/train.jsonl --val data/v4_mixed/val.jsonl --image-root data/v4_mixed --model openai/clip-vit-base-patch16 --epochs 10 --batch-size 4 --out checkpoints/v7_1_gen_q224.pt

python -m scripts.evaluate_generative --checkpoint checkpoints/v7_1_gen_q100.pt --data data/v4_hard/val.jsonl --image-root data/v4_hard
python -m scripts.evaluate_generative --checkpoint checkpoints/v7_1_gen_q224.pt --data data/v4_hard/val.jsonl --image-root data/v4_hard
```

Interpretation: if teacher-forced mean IoU is much higher than autoregressive mean IoU, error propagation is important; if both are poor, the model has not learned reliable image/text-conditioned coordinate prediction. **CE improvement alone is not grounding evidence.** All evaluation numbers require execution in the user's environment; no improvements are claimed in advance.

## V7: learn WHERE generative grounding comes from

**Goal:** Study how image-and-text inputs can be mapped to a *generated coordinate sequence*, as a small controlled analogue of the generative grounding capability in Qwen-VL. This is **not a Qwen-VL reproduction**, **does not load Qwen weights**, and does **not** claim to use the exact Qwen3-VL architecture or training recipe. Existing V1-V6 code and checkpoints remain untouched.

### What trains, and what is generated?

The same labeled image/text/GT JSONL used by V4-V6 is reused. A frozen CLIP ViT-B/16 extracts 196 visual patch tokens; frozen CLIP text encoder extracts a pooled text vector. Trainable projections form encoder memory; a small, trainable **causal Transformer Decoder** cross-attends to the memory and predicts 5 tokens:

```text
input:  image + "the yellow triangle"
GT:     [x1,y1,x2,y2] in normalized image coordinates
quant:  [round(1000*x1), round(1000*y1), round(1000*x2), round(1000*y2)]
BOS:    <BOS> (ID 1001) ; EOS: <EOS> (ID 1002)
training decoder input: [BOS, x1, y1, x2, y2]
training next tokens:  [x1, y1, x2, y2, EOS]
loss:   autoregressive cross-entropy (teacher forcing)
inference: starts from BOS, greedily predicts 4 discrete coordinates then EOS
```

Coordinate values are represented by **1001 discrete tokens** (0..1000), not literal JSON numerals. It is a compact didactic coordinate vocabulary, not Qwen's official tokenizer/serialization. During inference, the 4 coordinate positions are restricted to coordinate tokens; no GT boxes are used. The EOS prediction and invalid coordinate geometry are reported. Invalid boxes are **not repaired silently**, so their IoU is zero. Image/text are the **only model inference inputs**; GT is used to form teacher-forced targets during training.

Run on the same cleaned 800-example synthetic joint train / 60-example Mixed validation dataset:

```powershell
git pull origin main
python -m pytest -q
python -m scripts.train_generative --train data/v4_joint/train.jsonl --val data/v4_mixed/val.jsonl --image-root data/v4_mixed --model openai/clip-vit-base-patch16 --epochs 10 --batch-size 4 --out checkpoints/v7_generative_b16.pt
python -m scripts.evaluate_generative --checkpoint checkpoints/v7_generative_b16.pt --data data/v4_hard/val.jsonl --image-root data/v4_hard
python -m scripts.visualize_generative --checkpoint checkpoints/v7_generative_b16.pt --image data/v4_hard/val_0094.png --text "the yellow triangle" --output results/v7_generative_0094.png
```

**Study points:** teacher forcing vs self-generated history, discrete coordinate quantization, causal self-attention, decoder cross-attention to image/text features, token CE vs geometric IoU, coordinate sequence validity, and the gap between train token accuracy and generated box accuracy. The frozen encoder deliberately simplifies training on an 8GB-class GPU. Compare **mechanisms**, not just benchmark scores; to understand a real Qwen-VL system, later study its native visual encoder, token packing, multimodal LLM and grounding-specific supervised data.

## V6: multi-scale coarse-to-fine refinement (experimental)

**Research rationale.** V4 Spatial B/16 is the strongest measured Hard Top-1 baseline (80% Acc@0.5, 0.615 Mean IoU); V5 region (52%) and V5.1 anchor-free (66%) have not surpassed it. V6 **retains V4's trained Spatial Head** and learns a residual correction instead of replacing its predictor.

`--arch coarse_fine` first produces the **same V4 Spatial box**, then samples a 3x3 ROI from both the native CLIP patch map (14x14 for ViT-B/16) and its pooled 7x7 map. The two ROI features, language representation, and original box pass through a zero-initialized residual MLP to shift center and scale width/height. This is **two pooled scales of one final CLIP feature map**, not a true multi-layer FPN or a separate higher-resolution vision encoder. This first V6 implementation focuses on refinement, not recovering entirely missed objects; because it uses the V4 predicted ROI, large target-selection errors can persist.

The final residual layer initializes at zero: **before training**, a V6 model correctly initialized from V4 emits the original V4 boxes. `--freeze-coarse` trains only the refinement module; without it, the V4 coarse head can also change. Previous `spatial`, `region`, `region_af`, and `baseline` architectures remain compatible with their checkpoints.

```powershell
git pull origin main
python -m pytest -q
python -m scripts.train --arch coarse_fine --init-checkpoint checkpoints/v4_b16.pt --freeze-coarse --model openai/clip-vit-base-patch16 --loss giou --spatial-weight 0 --train data/v4_joint/train.jsonl --val data/v4_mixed/val.jsonl --image-root data/v4_mixed --epochs 10 --batch-size 4 --lr 1e-4 --out checkpoints/v6_refine_b16.pt
python -m scripts.evaluate --checkpoint checkpoints/v4_b16.pt --data data/v4_hard/val.jsonl --image-root data/v4_hard
python -m scripts.evaluate --checkpoint checkpoints/v6_refine_b16.pt --data data/v4_hard/val.jsonl --image-root data/v4_hard
python -m scripts.visualize --checkpoint checkpoints/v6_refine_b16.pt --image data/v4_hard/val_0094.png --text "the yellow triangle" --output results/v6_val0094.png --heatmap results/v6_val0094_heatmap.png
```

Check performance on **the same V4 Hard set**, and do not overwrite `checkpoints/v4_b16.pt`. A stronger V6-B later could integrate proposal-aware features or pretrained multilevel backbone features, after this ablation has been measured. New code has not yet been tested on the user's GPU; no V6 performance improvement is claimed.

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

## V4: dataset correctness and attribute-confusion tests

**Critical data fix:** Older generator revisions drew triangle masks with `fill=1` instead of `fill=255`. Their triangles were nearly transparent and could appear as another shape below them. This invalidates clean shape-grounding interpretations of prior V1–V3 runs. Do **not** compare new dataset numbers directly to old runs as though only the architecture changed. Archived results remain historical observations, not validated shape benchmarks.

The fixed generator draws opaque triangles, records `target_index`, `same_color_distractors`, `same_shape_distractors`, and prefers hard targets with both kinds of distractors when visible. The original full-object amodal boxes are retained.

Regenerate **new folders** without overwriting old datasets or checkpoints:

```powershell
git pull origin main
python -m pytest -q
python -m scripts.make_toy_data --mode mixed --out data/v4_mixed --train 300 --val 60
python -m scripts.make_toy_data --mode hard --out data/v4_hard --train 500 --val 100
```

Before retraining, visually inspect images showing triangles and audit the new JSONL labels; old checkpoint scores should not be used as new V4 performance numbers. Rebuild the joint training JSONL from `v4_mixed/train.jsonl` and `v4_hard/train.jsonl` only, preserving correct image paths, then train B/32 and B/16 with identical data, loss, seed and evaluation splits. This is a new dataset version, **not** a direct continuation of the previous benchmark.

## V5: patch-anchored region selection (experimental)

Historical V1–V4 code is preserved on [`archive/v4-stable`](https://github.com/engrenage0409/Visual-Grounding-Lab/tree/archive/v4-stable). In `main`, `--arch baseline` and `--arch spatial` continue to load prior checkpoints; new V5 functionality uses `--arch region`. Historical trained `.pt` checkpoints are **local files**, not stored in GitHub unless separately uploaded.

**Hypothesis:** V4's expected-center estimator can average two attention peaks into an implausible middle position. V5 assigns each CLIP patch an independent text-conditioned match logit and a candidate xyxy box. At inference, the highest-scored candidate is selected. During training, the cell containing the GT box center receives cross-entropy supervision, and that positive candidate receives box L1 + IoU/GIoU regression. This is a simple patch-anchored candidate selector, **not** full DETR, region proposal networks, or object-level negative mining. `--spatial-weight` controls selection CE weight for `region` and original spatial CE for `spatial`.

Run with the **fixed V4 joint data** (not older datasets containing nearly invisible triangles):

```powershell
git pull origin main
python -m pytest -q
python -m scripts.train --arch region --model openai/clip-vit-base-patch16 --loss giou --spatial-weight 1.0 --train data/v4_joint/train.jsonl --val data/v4_mixed/val.jsonl --image-root data/v4_mixed --epochs 10 --batch-size 4 --out checkpoints/v5_region_b16.pt
python -m scripts.evaluate --checkpoint checkpoints/v5_region_b16.pt --data data/v4_hard/val.jsonl --image-root data/v4_hard
python -m scripts.visualize --checkpoint checkpoints/v5_region_b16.pt --image data/v4_hard/val_0094.png --text "the yellow triangle" --output results/v5_val0094.png --heatmap results/v5_val0094_heatmap.png
```

`visualize` draws the selected candidate in red, additional top-ranked candidate boxes in orange, and an optional score heatmap. Compare against `checkpoints/v4_b16.pt` on **identical** V4 Mixed/Hard sets. V5 has not yet been GPU-trained or verified to outperform V4.

## V5.1: anchor-free region head and Oracle Top-K diagnostics

This is a **new architecture** `--arch region_af`; the old `baseline`, `spatial` (V4), and `region` (V5) code paths remain available and compatible with their existing head checkpoints. Do not overwrite historical `.pt` files.

`region_af` predicts a text-conditioned match logit, four L/T/R/B distances from each patch center, and an IoU-quality logit. In contrast to classical anchor-box detectors, no predefined box sizes or aspect ratios are used. GT-center sampling labels **multiple** interior points as positive, with a nearest-point fallback for tiny targets. The model trains all positive candidates with L1 + GIoU and uses balanced positive/negative BCE for matching plus soft IoU quality supervision. The inference score is `sigmoid(match) * sigmoid(quality)`, and inference selects one top-scored box. This is a single-scale FCOS-inspired research head, **not full YOLO or FCOS**. The `--spatial-weight` option controls the match/quality loss weight on `region_af`.

Train on the **fixed V4** dataset to compare against the existing V4 and V5 checkpoints:

```powershell
git pull origin main
python -m pytest -q
python -m scripts.train --arch region_af --model openai/clip-vit-base-patch16 --loss giou --spatial-weight 1.0 --train data/v4_joint/train.jsonl --val data/v4_mixed/val.jsonl --image-root data/v4_mixed --epochs 10 --batch-size 4 --out checkpoints/v5_af_b16.pt
python -m scripts.evaluate --checkpoint checkpoints/v5_af_b16.pt --data data/v4_hard/val.jsonl --image-root data/v4_hard
```

Diagnose candidate generation and ranking **before/after** using the same V4 Hard validation set:

```powershell
python -m scripts.diagnose_regions --checkpoint checkpoints/v5_region_b16.pt --data data/v4_hard/val.jsonl --image-root data/v4_hard --out results/v5_oracle.json
python -m scripts.diagnose_regions --checkpoint checkpoints/v5_af_b16.pt --data data/v4_hard/val.jsonl --image-root data/v4_hard --out results/v5_af_oracle.json
python -m scripts.visualize --checkpoint checkpoints/v5_af_b16.pt --image data/v4_hard/val_0094.png --text "the yellow triangle" --output results/v5_af_0094.png --heatmap results/v5_af_0094_heatmap.png
```

`Oracle@K` calculates the best GT IoU among the model's top-K candidates **using ground truth at evaluation time**. It is a diagnostic upper bound, **not deployable inference performance**. Compare actual Top-1 to Oracle@5 and Oracle@All to separate ranking from candidate-regression failures. The code is not yet GPU-trained or validated on the user's machine; do not claim it improves V4 until measured.

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
