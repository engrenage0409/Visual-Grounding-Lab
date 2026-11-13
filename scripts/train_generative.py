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
def evaluate(model, loader, device):
    model.eval()
    count, correct, iou_sum, invalid, eos_correct = 0, 0, 0., 0, 0
    for batch in loader:
        generated = model.generate(batch["images"], batch["texts"])
        boxes = generated["boxes"]
        truth = batch["boxes"].to(device)
        iou = box_iou_aligned(boxes, truth)
        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        count += len(iou)
        correct += int((iou >= .5).sum())
        iou_sum += float(iou.sum())
        invalid += int((~valid).sum())
        eos_correct += int(generated["eos_correct"].sum())
    return {
        "n": count,
        "acc_at_0.5": correct/count,
        "mean_iou": iou_sum/count,
        "invalid_box_rate": invalid/count,
        "eos_accuracy": eos_correct/count,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="data/v4_joint/train.jsonl")
    p.add_argument("--val", default="data/v4_mixed/val.jsonl")
    p.add_argument("--image-root", default="data/v4_mixed")
    p.add_argument("--model", default="openai/clip-vit-base-patch16")
    p.add_argument("--epochs", type=int, default=10)
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
                               heads=a.heads, dropout=a.dropout).to(device)
    optimizer = torch.optim.AdamW(
        (param for param in model.parameters() if param.requires_grad), lr=a.lr)
    output = Path(a.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    best = (-1., -1.)
    for epoch in range(1, a.epochs+1):
        model.train()
        losses, accs = [], []
        for batch in tqdm(train_loader, desc=f"V7 epoch {epoch}"):
            optimizer.zero_grad(set_to_none=True)
            result = model(batch["images"], batch["texts"], batch["boxes"].to(device))
            result["loss"].backward()
            optimizer.step()
            losses.append(float(result["loss"].detach()))
            accs.append(float(result["token_accuracy"]))
        scores = evaluate(model, val_loader, device)
        print(json.dumps({"epoch":epoch, "train_token_ce":sum(losses)/len(losses),
                          "train_teacher_forced_token_acc":sum(accs)/len(accs),
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
                "coord_max":1000, "output_length":5,
            }, output)
    print("Best checkpoint saved:", output, "device:",device)


if __name__ == "__main__":
    main()
