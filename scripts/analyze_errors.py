"""Per-sample visual grounding error analysis on V4 synthetic JSONL.

Example:
python -m scripts.analyze_errors --checkpoint checkpoints/v4_b16.pt --data data/v4_hard/val.jsonl --image-root data/v4_hard --out results/v4_b16_hard
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from grounding.checkpoints import load_grounder
from grounding.data import ReferringDataset, collate
from grounding.geometry import box_iou_aligned


def summarize(rows):
    if not rows:
        return {"n": 0, "acc_at_0.5": None, "mean_iou": None}
    return {
        "n": len(rows),
        "acc_at_0.5": round(sum(r["correct"] for r in rows) / len(rows), 4),
        "mean_iou": round(sum(r["iou"] for r in rows) / len(rows), 4),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--image-root", required=True)
    p.add_argument("--out", default="results/error_analysis")
    p.add_argument("--batch-size", type=int, default=4)
    a = p.parse_args()

    records = [json.loads(line) for line in Path(a.data).read_text(encoding="utf-8").splitlines() if line.strip()]
    dataset = ReferringDataset(a.data, a.image_root)
    loader = DataLoader(dataset, batch_size=a.batch_size, shuffle=False, collate_fn=collate, num_workers=0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ck = load_grounder(a.checkpoint, device)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    results = []
    cursor = 0
    with torch.inference_mode():
        for batch in loader:
            pred = model(batch["images"], batch["texts"])
            gt = batch["boxes"].to(device)
            ious = box_iou_aligned(pred, gt).cpu().tolist()
            for j, iou in enumerate(ious):
                row = records[cursor + j]
                objects = row.get("objects", [])
                expression = row["expression"]
                target_index = row.get("target_index")
                if target_index is None:
                    matches = [k for k, obj in enumerate(objects) if expression == f'the {obj["color"]} {obj["shape"]}']
                    target_index = matches[0] if len(matches) == 1 else None
                target = objects[target_index] if target_index is not None else {}
                color = target.get("color", "")
                shape = target.get("shape", "")
                same_color = sum(k != target_index and obj.get("color") == color for k, obj in enumerate(objects)) if target else 0
                same_shape = sum(k != target_index and obj.get("shape") == shape for k, obj in enumerate(objects)) if target else 0

                image = batch["images"][j]
                w, h = image.size
                box = row["bbox_xyxy"]
                area_fraction = ((box[2] - box[0]) * (box[3] - box[1])) / (w * h)
                visible = row.get("target_visible_fraction", 1.0)
                pred_px = (pred[j].detach().cpu() * torch.tensor([w,h,w,h])).tolist()
                results.append({
                    "image": str(batch["paths"][j]),
                    "expression": expression,
                    "color": color,
                    "shape": shape,
                    "iou": round(float(iou), 6),
                    "correct": int(iou >= 0.5),
                    "same_color_count": same_color,
                    "same_shape_count": same_shape,
                    "same_color": int(same_color > 0),
                    "same_shape": int(same_shape > 0),
                    "visible_fraction": visible,
                    "occlusion_group": "occluded" if visible < 0.99 else "fully_visible",
                    "size_fraction": round(area_fraction, 6),
                    "size_group": "small" if area_fraction < 0.04 else ("medium" if area_fraction < 0.09 else "large"),
                    "gt_xyxy": json.dumps(box),
                    "pred_xyxy": json.dumps([round(x, 2) for x in pred_px]),
                })
            cursor += len(ious)

    if cursor != len(records):
        raise ValueError("Dataset/record count mismatch")

    fields = list(results[0].keys())
    with (out / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    groups = {}
    for key in ("same_color", "same_shape", "occlusion_group", "size_group", "shape"):
        buckets = defaultdict(list)
        for r in results:
            buckets[str(r[key])].append(r)
        groups[key] = {k: summarize(v) for k, v in sorted(buckets.items())}

    report = {
        "checkpoint": a.checkpoint,
        "model_name": ck.get("model_name"),
        "dataset": a.data,
        "overall": summarize(results),
        "groups": groups,
        "worst_cases": [
            {k: r[k] for k in ("image", "expression", "iou", "gt_xyxy", "pred_xyxy", "visible_fraction", "same_color_count", "same_shape_count")}
            for r in sorted(results, key=lambda r: r["iou"])[:10]
        ],
    }
    (out / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved: {out / 'predictions.csv'} and {out / 'summary.json'}")
    print("Groups overlap; do not add their counts across different dimensions.")


if __name__ == "__main__":
    main()
