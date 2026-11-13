"""Evaluate V7 generative boxes on the same JSONL sets as V4-V6."""
import argparse
import json
import torch
from torch.utils.data import DataLoader
from grounding.data import ReferringDataset, collate
from grounding.generative_checkpoints import load_generative
from scripts.train_generative import evaluate


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",default="checkpoints/v7_generative_b16.pt")
    p.add_argument("--data",default="data/v4_hard/val.jsonl")
    p.add_argument("--image-root",default="data/v4_hard")
    p.add_argument("--batch-size",type=int,default=4)
    p.add_argument("--constrain-box",action="store_true",help="Constrain generated x2>x1, y2>y1")
    a=p.parse_args()
    device="cuda" if torch.cuda.is_available() else "cpu"
    model,ck=load_generative(a.checkpoint,device)
    loader=DataLoader(ReferringDataset(a.data,a.image_root),
                       batch_size=a.batch_size,shuffle=False,collate_fn=collate,num_workers=0)
    print(json.dumps({"arch":ck["arch"],"checkpoint_epoch":ck["epoch"],
                      **evaluate(model,loader,device,constrain_box=a.constrain_box)},indent=2))


if __name__=="__main__":
    main()
