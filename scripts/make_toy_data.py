"""Synthetic referring-expression dataset with controlled difficulty.

Easy: legacy-style rectangles; Mixed: randomized shape/color/position/size;
Hard: more distractors, lookalikes, partial overlaps, and visibility checks.
Bounding boxes are full object (amodal) boxes; occlusion metadata is explicit.
All samples remain synthetic smoke tests, not a RefCOCO benchmark.
"""
import argparse
import json
import math
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageChops

PALETTE = {
    "red": (226, 61, 67),
    "blue": (55, 110, 224),
    "green": (51, 174, 91),
    "yellow": (236, 195, 48),
    "purple": (153, 94, 205),
    "orange": (242, 139, 51),
}
SHAPES = ("rectangle", "circle", "triangle")
SIZE = 224

def shape_mask(shape, box):
    mask = Image.new("L", (SIZE, SIZE), 0)
    draw = ImageDraw.Draw(mask)
    x0, y0, x1, y1 = box
    if shape == "rectangle":
        draw.rectangle(box, fill=255)
    elif shape == "circle":
        draw.ellipse(box, fill=255)
    elif shape == "triangle":
        draw.polygon([(int((x0+x1)/2), y0), (x0, y1), (x1, y1)], fill=255)
    else:
        raise ValueError(f"Unknown shape: {shape}")
    return mask

def intersection_over_union(a, b):
    x0 = max(a[0], b[0]); y0 = max(a[1], b[1])
    x1 = min(a[2], b[2]); y1 = min(a[3], b[3])
    inter = max(0, x1-x0)*max(0, y1-y0)
    area_a = (a[2]-a[0])*(a[3]-a[1])
    area_b = (b[2]-b[0])*(b[3]-b[1])
    return inter / max(area_a+area_b-inter, 1)

def sample_box(rng, mode, existing):
    for _ in range(250):
        side_x = rng.randint(28 if mode == "hard" else 32, 64)
        side_y = rng.randint(28 if mode == "hard" else 32, 67)
        x = rng.randint(8, SIZE-side_x-9)
        y = rng.randint(8, SIZE-side_y-9)
        box = [x, y, x+side_x, y+side_y]
        max_iou = max((intersection_over_union(box, other) for other in existing), default=0)
        if mode == "mixed" and max_iou <= 0.03:
            return box
        if mode == "hard" and max_iou <= 0.24:
            return box
    return box

def make_scene(rng, mode):
    image = Image.new("RGB", (SIZE, SIZE), (246, 246, 242))
    if mode == "easy":
        # Keep as a simple control, but randomize color order to remove position shortcut.
        colors = rng.sample(list(PALETTE)[:3], 3)
        objects = []
        for j, color in enumerate(colors):
            box = [15+j*70+rng.randint(-4, 4), rng.randint(55, 135), 0, 0]
            box[2] = box[0]+45
            box[3] = box[1]+50
            objects.append({"color": color, "shape": "rectangle", "bbox_xyxy": box})
    else:
        n = rng.randint(3, 5) if mode == "mixed" else rng.randint(5, 7)
        # Each color/shape combination is unique: expression identifies one object.
        combinations = [(c, s) for c in PALETTE for s in SHAPES]
        # Sample distinct attribute pairs; hard cases force color/shape lookalikes.
        if mode == "hard":
            chosen_color, chosen_shape = rng.choice(combinations)
            same_color = (chosen_color, rng.choice([s for s in SHAPES if s != chosen_shape]))
            same_shape = (rng.choice([c for c in PALETTE if c != chosen_color]), chosen_shape)
            forced = [(chosen_color, chosen_shape), same_color, same_shape]
            remaining = [pair for pair in combinations if pair not in forced]
            pairs = forced + rng.sample(remaining, n-len(forced))
        else:
            pairs = rng.sample(combinations, n)
        objects = []
        for color, shape in pairs:
            box = sample_box(rng, mode, [o["bbox_xyxy"] for o in objects])
            objects.append({"color": color, "shape": shape, "bbox_xyxy": box})
        if mode == "hard":
            # Deliberately create some close objects, while avoiding total cover.
            if rng.random() < 0.65:
                anchor = objects[0]["bbox_xyxy"]
                w = objects[-1]["bbox_xyxy"][2] - objects[-1]["bbox_xyxy"][0]
                h = objects[-1]["bbox_xyxy"][3] - objects[-1]["bbox_xyxy"][1]
                x = max(8, min(SIZE-w-8, anchor[0]+rng.randint(14, 30)))
                y = max(8, min(SIZE-h-8, anchor[1]+rng.randint(14, 30)))
                objects[-1]["bbox_xyxy"] = [x, y, x+w, y+h]
    masks = []
    for obj in objects:
        mask = shape_mask(obj["shape"], obj["bbox_xyxy"])
        image.paste(PALETTE[obj["color"]], (0, 0, SIZE, SIZE), mask)
        masks.append(mask)
    # Compute visible fraction with later layers covering earlier layers.
    visible = []
    later_coverage = Image.new("L", (SIZE, SIZE), 0)
    for mask in reversed(masks):
        total = mask.getbbox()
        original_area = sum(mask.get_flattened_data())
        still_visible = ImageChops.subtract(mask, later_coverage)
        ratio = sum(still_visible.get_flattened_data())/max(1, original_area)
        visible.append(ratio)
        later_coverage = ImageChops.lighter(later_coverage, mask)
    visible.reverse()
    options = [i for i, ratio in enumerate(visible) if ratio >= 0.68]
    # In hard mode, prefer a target that has both a same-color and a same-shape distractor.
    # This makes attribute composition measurable rather than accidental.
    if mode == "hard":
        difficult = [i for i in options if
                     any(j != i and o["color"] == objects[i]["color"] for j,o in enumerate(objects)) and
                     any(j != i and o["shape"] == objects[i]["shape"] for j,o in enumerate(objects))]
        target_idx = rng.choice(difficult or options) if options else len(objects)-1
    else:
        target_idx = rng.choice(options) if options else len(objects)-1
    target = objects[target_idx]
    label = {
        "expression": f'the {target["color"]} {target["shape"]}',
        "bbox_xyxy": target["bbox_xyxy"],
        "target_visible_fraction": round(visible[target_idx], 3),
        "bbox_type": "amodal",
        "difficulty": mode,
        "same_color_distractors": sum(o["color"] == target["color"] and i != target_idx for i,o in enumerate(objects)),
        "same_shape_distractors": sum(o["shape"] == target["shape"] and i != target_idx for i,o in enumerate(objects)),
        "target_index": target_idx,
        "objects": objects,
    }
    return image, label

def generate(out="data/toy", train=80, val=20, mode="mixed", seed=42):
    if mode not in ("easy", "mixed", "hard"):
        raise ValueError("mode must be easy, mixed or hard")
    if train < 1 or val < 1:
        raise ValueError("train and val must both be positive")
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    for split, count, offset in (("train", train, 0), ("val", val, 10_000_000)):
        rng = random.Random(seed+offset)
        with (root/f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for i in range(count):
                image, label = make_scene(rng, mode)
                name = f"{split}_{i:04d}.png"
                image.save(root/name)
                f.write(json.dumps({"image": name, **label}, ensure_ascii=False)+"\n")
    return root

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/toy")
    p.add_argument("--mode", choices=("easy", "mixed", "hard"), default="mixed")
    p.add_argument("--train", type=int, default=80)
    p.add_argument("--val", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    print("Generated", generate(a.out, a.train, a.val, a.mode, a.seed), "mode=", a.mode)

if __name__ == "__main__":
    main()
