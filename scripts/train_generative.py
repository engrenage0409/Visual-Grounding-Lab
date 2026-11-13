"""Train V7 generative coordinate decoder on existing JSONL (image,text,xyxy)."""
import argparse
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from grounding.data import ReferringDataset, collate
from grounding.generative import GenerativeGrounder
from grounding.geometry import box_iou_aligned


@torch.no_grad()
def evaluate(model, loader, device, constrain_box=False):
    """Report actual autoregressive metrics separately from teacher-forced ones."""
    model.eval()
    count = correct = invalid = eos_correct = 0
    tf_correct = tf_invalid = coord_exact = coord_total = 0
    iou_sum = tf_iou_sum = coord_error_sum = 0.
    for batch in loader:
        truth = batch["boxes"].to(device)
        gen = model.generate(batch["images"], batch["texts"],
                             constrain_box=constrain_box)
        boxes = gen["boxes"]
        iou = box_iou_aligned(boxes, truth)
        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        # Teacher forcing is an *upper-information* diagnostic:
        # previous GT tokens are supplied to each next-token prediction.
        tf = model(batch["images"], batch["texts"], truth)
        tf_boxes = tf["teacher_forced_boxes"]
        tf_iou = box_iou_aligned(tf_boxes, truth)
        tf_valid = (tf_boxes[:, 2] > tf_boxes[:, 0]) & (tf_boxes[:, 3] > tf_boxes[:, 1])
        target_tokens = (truth.clamp(0,1)*model.coord_max).round().long()
        generated_tokens = gen["tokens"][:, :4]
        n = len(iou)
        count += n
        correct += int((iou >= .5).sum().item())
        iou_sum += float(iou.sum().item())
        invalid += int((~valid).sum().item())
        eos_correct += int(gen["eos_correct"].sum().item())
        tf_correct += int((tf_iou >= .5).sum().item())
        tf_iou_sum += float(tf_iou.sum().item())
        tf_invalid += int((~tf_valid).sum().item())
        coord_exact += int((generated_tokens == target_tokens).sum().item())
        coord_total += 4*n
        coord_error_sum += float((boxes-truth).abs().sum().item())
    if not count:
        raise ValueError("Empty evaluation dataset")
    return {
        "n":count, "acc_at_0.5":correct/count,
        "mean_iou":iou_sum/count,
        "invalid_box_rate":invalid/count,
        "eos_accuracy":eos_correct/count,
        "generated_coordinate_token_acc":coord_exact/coord_total,
        "generated_coordinate_mae_normalized":coord_error_sum/coord_total,
        "teacher_forced_acc_at_0.5":tf_correct/count,
        "teacher_forced_mean_iou":tf_iou_sum/count,
        "teacher_forced_invalid_box_rate":tf_invalid/count,
        "constrain_box":constrain_box,
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="data/v4_joint/train.jsonl")
    p.add_argument("--val", default="data/v4_mixed/val.jsonl")
    p.add_argument("--image-root", default="data/v4_mixed")
    p.add_argument("--model", default="openai/clip-vit-base-patch16")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--coord-max", type=int, choices=[100,224,1000], default=1000,
                   help="Coordinate vocabulary resolution; compare independent runs")
    p.add_argument("--constrain-box", action="store_true",
                   help="Validation decode only: enforce x2>x1 and y2>y1")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--out", default="checkpoints/v7_generative_b16.pt")
    a = p.parse_args()
    torch.manual_seed(42)
    random.seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader = DataLoader(ReferringDataset(a.train, a.image_root),
                              batch_size=a.batch_size, shuffle=True, collate_fn=collate,
                              num_workers=0)
    val_loader = DataLoader(ReferringDataset(a.val, a.image_root),
                            batch_size=a.batch_size, shuffle=False, collate_fn=collate,
                            num_workers=0)
    model = GenerativeGrounder(a.model, hidden=a.hidden, layers=a.layers,
                               heads=a.heads, dropout=a.dropout,
                               coord_max=a.coord_max).to(device)
    optimizer = torch.optim.AdamW(
        (param for param in model.parameters() if param.requires_grad), lr=a.lr)
    output = Path(a.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    best = (-1., -1.)
    for epoch in range(1, a.epochs+1):
        model.train()
        losses, accs, coord_accs, eos_accs = [], [], [], []
        for batch in tqdm(train_loader, desc=f"V7 epoch {epoch}"):
            optimizer.zero_grad(set_to_none=True)
            result = model(batch["images"], batch["texts"], batch["boxes"].to(device))
            result["loss"].backward()
            optimizer.step()
            losses.append(float(result["loss"].detach()))
            accs.append(float(result["token_accuracy"]))
            coord_accs.append(float(result["coordinate_token_accuracy"]))
            eos_accs.append(float(result["eos_token_accuracy"]))
        scores = evaluate(model, val_loader, device,
                          constrain_box=a.constrain_box)
        print(json.dumps({"epoch":epoch, "train_token_ce":sum(losses)/len(losses),
                          "train_teacher_forced_token_acc":sum(accs)/len(accs),
                          "train_coordinate_token_acc":sum(coord_accs)/len(coord_accs),
                          "train_eos_token_acc":sum(eos_accs)/len(eos_accs),
                          **scores}))
        current = (scores["acc_at_0.5"], scores["mean_iou"])
        if current > best:
            best = current
            head = {key:value.detach().cpu() for key,value in model.state_dict().items()
                    if not key.startswith("clip.")}
            torch.save({
                "arch":"generative", "model_name":a.model, "hidden":a.hidden,
                "layers":a.layers, "heads":a.heads, "dropout":a.dropout,
                "head":head, "epoch":epoch, "metrics":scores,
                "coord_max":a.coord_max, "output_length":5,
                "constrain_box":a.constrain_box,
            }, output)
    print("Best checkpoint saved:", output, "device:",device)


if __name__ == "__main__":
    main()
